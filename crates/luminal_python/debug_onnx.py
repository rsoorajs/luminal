import os
import sys
import tempfile
import torch
from tests.test_models import CastDoubleToFloatModel, CastInt32ToFloatModel
import onnx

def inspect_onnx(model, x, name):
    """Export model to ONNX and inspect the graph."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model_gm = torch.export.export(model, (x,), strict=False).graph_module
        torch.onnx.export(
            model_gm,
            (x,),
            tmp_path,
            input_names=["input_0"],
        )

        # Load and inspect the ONNX model
        onnx_model = onnx.load(tmp_path)
        print(f"\n=== ONNX Graph for {name} ===")
        print(f"Input tensor dtype: {x.dtype}")

        # Print inputs
        print("\nInputs:")
        for inp in onnx_model.graph.input:
            if inp.type.tensor_type:
                elem_type = inp.type.tensor_type.elem_type
                print(f"  {inp.name}: elem_type={elem_type}")

        # Print nodes
        print("\nNodes:")
        for node in onnx_model.graph.node:
            print(f"  {node.op_type}: {node.input} -> {node.output}")
            if node.op_type == "Cast":
                for attr in node.attribute:
                    print(f"    {attr.name}={attr.i}")

        # Print outputs
        print("\nOutputs:")
        for out in onnx_model.graph.output:
            if out.type.tensor_type:
                elem_type = out.type.tensor_type.elem_type
                print(f"  {out.name}: elem_type={elem_type}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Test 1: float64 -> float32 cast
print("\n" + "="*60)
print("TEST 1: Fresh PyTorch state")
print("="*60)
model1 = CastDoubleToFloatModel()
x1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
inspect_onnx(model1, x1, "CastDoubleToFloat")

# Test 2: int32 -> float32 cast
print("\n" + "="*60)
print("TEST 2: After previous export")
print("="*60)
model2 = CastInt32ToFloatModel()
x2 = torch.tensor([1, 2], dtype=torch.int32)
inspect_onnx(model2, x2, "CastInt32ToFloat")
