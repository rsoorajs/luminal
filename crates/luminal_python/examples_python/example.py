"""Example: using luminal backends with torch.compile.

Usage:
    # Auto-detect backend (picks cuda_lite if GPU available, native otherwise):
    python example.py

    # Use an external backend plugin (e.g. luminal_cuda for cuda_heavy):
    python example.py --backend cuda_heavy
"""

import argparse

import torch
import luminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, help="'cuda_heavy' to use luminal_cuda plugin")
    args = parser.parse_args()

    if args.backend == "cuda_heavy":
        import luminal_cuda
        backend = luminal.register_backend(luminal_cuda.luminal_backend)
    else:
        backend = luminal.luminal_backend

    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )

    # Move to GPU if a CUDA backend is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Compile with luminal
    compiled = torch.compile(model, backend=backend)

    # Run inference
    x = torch.randn(4, 128, device=device)
    out = compiled(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
