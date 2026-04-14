"""Example: using an external backend plugin with luminal.

External backends auto-register via Python entry_points when you
`import luminal`. Just install the backend package (e.g. `pip install
luminal-walrus`) and it will appear in `available_backends()`.

Usage:
    # Use the default backend (picks cuda if available, else native):
    python example.py

    # Force a specific backend via environment variable:
    LUMINAL_BACKEND=cuda_heavy python example.py
"""

import torch
import luminal

# Show which backends are registered.
# Built-in: "native", "cpu", and (if compiled with cuda feature) "cuda", "cuda_lite", "gpu"
# Plugins add their own names here automatically.
print("Available backends:", luminal.available_backends())

# Define a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
)

# Move to GPU if a CUDA backend is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Compile with luminal — the backend is selected by LUMINAL_BACKEND env var
# (defaults to "cuda" if available, "native" otherwise)
compiled = torch.compile(model, backend=luminal.luminal_backend)

# Run inference
x = torch.randn(4, 128, device=device)
out = compiled(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Backend used: {compiled._luminal_graph.backend if hasattr(compiled, '_luminal_graph') else 'unknown'}")
