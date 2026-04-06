"""Built-in shape presets for testing and benchmarking."""

from typing import TypeAlias

ShapePreset: TypeAlias = list[tuple[int, ...]]

# Common LLM dimensions
STANDARD_SHAPES: ShapePreset = [
    (128,), (256,), (512,), (1024,), (2048,), (4096,), (8192,),
    (128, 128), (256, 256), (512, 512), (1024, 1024),
    (4096, 4096), (8192, 8192),
]

# Non-power-of-2, primes, edge cases
EDGE_SHAPES: ShapePreset = [
    (1,), (3,), (7,), (17,), (127,), (255,), (1000,),
    (1, 1), (1, 4096), (4096, 1), (13, 17),
    (33, 65), (1023, 1025),
]

# Typical Llama / GPT linear layer shapes (M, N)
LLM_SHAPES: ShapePreset = [
    (1, 4096),          # single-token decode
    (32, 4096),         # small batch decode
    (128, 4096),        # medium batch
    (512, 4096),        # prefill chunk
    (2048, 4096),       # long prefill
    (4096, 4096),       # square
    (4096, 11008),      # Llama MLP up-projection
    (4096, 14336),      # Llama-2 70B MLP
    (11008, 4096),      # Llama MLP down-projection
]
