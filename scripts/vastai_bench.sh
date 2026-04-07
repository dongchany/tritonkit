#!/bin/bash
# vastai_bench.sh — Spin up vast.ai instance, run tritonkit benchmarks, destroy
#
# CRITICAL: Always destroys the instance on exit (success OR failure OR interrupt)
# Usage:
#   ./scripts/vastai_bench.sh <gpu_filter> <gpu_label>
# Example:
#   ./scripts/vastai_bench.sh "gpu_name=RTX_4090" rtx_4090
#   ./scripts/vastai_bench.sh "gpu_name=H100_SXM" h100_sxm

set -uo pipefail

# ==================== CONFIG ====================
GPU_FILTER="${1:-gpu_name=RTX_4090}"
GPU_LABEL="${2:-unknown_gpu}"
MAX_PRICE_PER_HOUR="${MAX_PRICE:-2.0}"  # USD/hour cap, override with MAX_PRICE=X
DOCKER_IMAGE="${DOCKER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
DISK_GB="${DISK_GB:-30}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/benchmarks/results/vastai/$GPU_LABEL}"
LOG_DIR="${LOG_DIR:-$(pwd)/benchmarks/results/vastai/logs}"

VASTAI="$(pwd)/.venv/bin/vastai"
INSTANCE_ID=""

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${GPU_LABEL}_$(date -u +%Y%m%dT%H%M%SZ).log"

log() {
    echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

# ==================== DESTROY GUARD ====================
destroy_instance() {
    if [[ -n "$INSTANCE_ID" ]]; then
        log "DESTROYING instance $INSTANCE_ID (cost stops now)"
        $VASTAI destroy instance "$INSTANCE_ID" 2>&1 | tee -a "$LOG_FILE" || true
        # Verify destroyed
        sleep 3
        if $VASTAI show instance "$INSTANCE_ID" 2>/dev/null | grep -q "$INSTANCE_ID"; then
            log "WARNING: instance $INSTANCE_ID may still exist! Check vast.ai console."
        else
            log "Confirmed: instance $INSTANCE_ID destroyed"
        fi
    fi
}

trap destroy_instance EXIT INT TERM HUP

# ==================== PRE-FLIGHT ====================
log "=== TritonKit vast.ai benchmark run ==="
log "GPU filter: $GPU_FILTER"
log "GPU label: $GPU_LABEL"
log "Max price: \$$MAX_PRICE_PER_HOUR/hr"
log "Docker image: $DOCKER_IMAGE"
log "Results dir: $RESULTS_DIR"

if [[ ! -f "$SSH_KEY.pub" ]]; then
    log "ERROR: SSH key $SSH_KEY.pub not found. Generate with: ssh-keygen -t ed25519"
    exit 1
fi

# Verify vastai auth
if ! $VASTAI show user 2>&1 | grep -q "id"; then
    log "ERROR: vastai not authenticated. Run: vastai set api-key <KEY>"
    exit 1
fi

CREDIT=$($VASTAI show user 2>&1 | grep -i credit | awk '{print $NF}')
log "Account credit: \$$CREDIT"

# ==================== SEARCH FOR OFFER ====================
log "Searching for offers matching: $GPU_FILTER, max \$$MAX_PRICE_PER_HOUR/hr"

OFFER_JSON=$($VASTAI search offers "$GPU_FILTER verified=true rentable=true dph_total<$MAX_PRICE_PER_HOUR cuda_max_good>=12.0 disk_space>=$DISK_GB inet_down>=100" \
    --order "dph_total" --raw 2>&1 | head -200)

OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    if not data:
        print('NONE')
    else:
        # Take cheapest verified offer
        cheapest = data[0]
        print(cheapest['id'])
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    print('NONE')
")

if [[ "$OFFER_ID" == "NONE" || -z "$OFFER_ID" ]]; then
    log "ERROR: No matching offers found"
    exit 1
fi

OFFER_PRICE=$(echo "$OFFER_JSON" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
print(f\"{data[0]['dph_total']:.3f}\")
")

log "Selected offer: $OFFER_ID at \$$OFFER_PRICE/hr"

# ==================== CREATE INSTANCE ====================
log "Creating instance..."
CREATE_OUTPUT=$($VASTAI create instance "$OFFER_ID" \
    --image "$DOCKER_IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --raw 2>&1)

INSTANCE_ID=$(echo "$CREATE_OUTPUT" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('new_contract', ''))
except Exception:
    print('')
")

if [[ -z "$INSTANCE_ID" ]]; then
    log "ERROR: Failed to create instance"
    log "$CREATE_OUTPUT"
    exit 1
fi

log "Instance created: $INSTANCE_ID"

# ==================== ATTACH SSH KEY ====================
log "Attaching SSH key..."
$VASTAI attach ssh "$INSTANCE_ID" "$(cat $SSH_KEY.pub)" 2>&1 | tee -a "$LOG_FILE" || true

# ==================== WAIT FOR READY ====================
log "Waiting for instance to be ready..."
WAIT_COUNT=0
MAX_WAIT=60  # 5 minutes
while [[ $WAIT_COUNT -lt $MAX_WAIT ]]; do
    STATUS=$($VASTAI show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('actual_status', 'unknown'))
except Exception:
    print('error')
")
    log "  status: $STATUS"
    if [[ "$STATUS" == "running" ]]; then
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [[ "$STATUS" != "running" ]]; then
    log "ERROR: Instance never became ready"
    exit 1
fi

# ==================== GET SSH URL ====================
SSH_URL=$($VASTAI ssh-url "$INSTANCE_ID" 2>&1 | tail -1)
SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f1 | cut -d@ -f2)
SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f2)
SSH_USER=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d@ -f1)

log "SSH: $SSH_USER@$SSH_HOST:$SSH_PORT"

# Wait for SSH to be reachable
log "Waiting for SSH to be reachable..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "echo ssh_ok" 2>/dev/null | grep -q ssh_ok; then
        log "SSH ready"
        break
    fi
    sleep 5
done

# ==================== RUN BENCHMARK ====================
log "Cloning tritonkit and running benchmarks..."

REMOTE_SCRIPT='
set -e
cd /workspace
git clone https://github.com/dongchany/tritonkit.git || (cd tritonkit && git pull)
cd tritonkit
pip install --quiet uv 2>&1 | tail -5
pip install --quiet -e . 2>&1 | tail -5
# Optional baselines
pip install --quiet flag_gems gemlite 2>&1 | tail -5 || true
pip install --quiet xformers --no-deps 2>&1 | tail -5 || true
pip install --quiet bitsandbytes 2>&1 | tail -5 || true
pip install --quiet liger-kernel 2>&1 | tail -5 || true

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,memory.total --format=csv
python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
python -c "import triton; print(\"triton\", triton.__version__)"

mkdir -p benchmarks/results
python benchmarks/run_all.py 2>&1 | tail -30
ls -la benchmarks/results/
'

ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "$REMOTE_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

# ==================== PULL RESULTS ====================
log "Pulling results back..."
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" -i "$SSH_KEY" \
    "$SSH_USER@$SSH_HOST:/workspace/tritonkit/benchmarks/results/*.json" \
    "$RESULTS_DIR/" 2>&1 | tee -a "$LOG_FILE"

ls -la "$RESULTS_DIR/" | tee -a "$LOG_FILE"

# ==================== DESTROY (via trap) ====================
log "Benchmark complete. Trap will destroy instance now."
exit 0
