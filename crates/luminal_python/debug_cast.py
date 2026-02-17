#!/usr/bin/env python3
"""Debug script to examine the ONNX graph for Cast operations."""

import torch
import tempfile
from pathlib import Path

# Simple model that does a cast
class CastModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)

# Export to ONNX
model = CastModel()
x = torch.tensor([42.123456], dtype=torch.float64)

with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = Path(tmpdir) / "cast_model.onnx"

    # Export using torch.onnx
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["x"],
        output_names=["output"],
        opset_version=17
    )

    # Read and print the ONNX file
    import onnx
    onnx_model = onnx.load(str(onnx_path))

    print("=" * 80)
    print("ONNX Graph for Cast Model:")
    print("=" * 80)
    print(onnx_model.graph)

    print("\n" + "=" * 80)
    print("Node Details:")
    print("=" * 80)
    for i, node in enumerate(onnx_model.graph.node):
        print(f"\nNode {i}: {node.op_type}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        if node.attribute:
            print(f"  Attributes:")
            for attr in node.attribute:
                print(f"    {attr.name}: {attr}")
