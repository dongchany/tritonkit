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

CREDIT=$($VASTAI show user --raw 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(f\"{d.get('credit', 0):.2f}\")
except Exception:
    print('?')
")
log "Account credit: \$$CREDIT"

# ==================== SEARCH FOR OFFER ====================
log "Searching for offers matching: $GPU_FILTER, max \$$MAX_PRICE_PER_HOUR/hr"

# Use --raw to get full JSON, write to tempfile to avoid truncation
OFFER_JSON_FILE=$(mktemp)
$VASTAI search offers "$GPU_FILTER verified=true rentable=true dph_total<$MAX_PRICE_PER_HOUR cuda_max_good>=12.6 disk_space>=$DISK_GB inet_down>=100 num_gpus=1" \
    --order "dph_total" --raw > "$OFFER_JSON_FILE" 2>&1

PARSE_RESULT=$(python3 -c "
import json
with open('$OFFER_JSON_FILE') as f:
    try:
        data = json.load(f)
    except Exception as e:
        print(f'NONE NONE PARSE_ERROR_{e}')
        exit(0)
if not data:
    print('NONE NONE EMPTY')
else:
    o = data[0]
    print(f\"{o['id']} {o['dph_total']:.4f} OK\")
")

OFFER_ID=$(echo "$PARSE_RESULT" | awk '{print $1}')
OFFER_PRICE=$(echo "$PARSE_RESULT" | awk '{print $2}')
PARSE_STATUS=$(echo "$PARSE_RESULT" | awk '{print $3}')
rm -f "$OFFER_JSON_FILE"

if [[ "$OFFER_ID" == "NONE" || -z "$OFFER_ID" ]]; then
    log "ERROR: No matching offers found ($PARSE_STATUS)"
    exit 1
fi

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
[ -d tritonkit ] && rm -rf tritonkit
git clone https://github.com/dongchany/tritonkit.git
cd tritonkit

# Use whatever torch is in the Docker image; only install missing deps
echo "=== checking existing torch ==="
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), \"CUDA not available\""

# Install tritonkit without deps to avoid torch upgrade
pip install --quiet --no-deps -e . 2>&1 | tail -3
pip install --quiet tabulate 2>&1 | tail -3
# Triton might be in image; install if missing
python -c "import triton" 2>/dev/null || pip install --quiet triton 2>&1 | tail -3

# Optional baselines (best effort, all skip on failure)
pip install --quiet --no-deps flag_gems 2>&1 | tail -3 || true
pip install --quiet --no-deps gemlite 2>&1 | tail -3 || true
pip install --quiet --no-deps xformers 2>&1 | tail -3 || true
pip install --quiet --no-deps liger-kernel 2>&1 | tail -3 || true

echo "=== environment ==="
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,memory.total --format=csv,noheader
python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
python -c "import triton; print(\"triton\", triton.__version__)"

echo "=== running benchmarks ==="
mkdir -p benchmarks/results
python benchmarks/run_all.py 2>&1 | tail -40
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
