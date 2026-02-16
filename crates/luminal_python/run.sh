#!/bin/bash
set -e

echo "Building Rust extension with maturin..."
uv run maturin develop --manifest-path rust/Cargo.toml --release

echo "Running main.py..."
uv run python -m luminal.main
