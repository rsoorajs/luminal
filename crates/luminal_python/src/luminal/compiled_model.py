"""CompiledModel wrapper for the Rust OnnxGraphResult."""

from typing import List, Set

import numpy as np
import torch


class CompiledModel:
    """Wrapper around OnnxGraphResult that handles PyTorch tensor conversion."""

    def __init__(self, graph_result):
        """Initialize with a compiled OnnxGraphResult from Rust.

        Args:
            graph_result: The OnnxGraphResult from luminal_python.process_onnx()
        """
        self._graph = graph_result
        self._input_names = graph_result.input_names
        self._output_names = graph_result.output_names
        self._output_shapes = graph_result.output_shapes

        # Find all tensor names to detect _kn (transposed) variants
        all_tensor_names: Set[str] = set(graph_result.tensor_names)
        self._kn_inputs = {}  # Maps input name to its _kn variant name
        for name in self._input_names:
            kn_name = f"{name}_kn"
            if kn_name in all_tensor_names:
                self._kn_inputs[name] = kn_name

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Execute the compiled model with PyTorch tensor inputs.

        Args:
            *inputs: PyTorch tensors matching the model's input signature

        Returns:
            List of PyTorch tensors containing the model outputs
        """
        if len(inputs) != len(self._input_names):
            raise ValueError(
                f"Expected {len(self._input_names)} inputs, got {len(inputs)}"
            )

        input_device = inputs[0].device if inputs else torch.device("cpu")

        # Set input data
        for name, tensor in zip(self._input_names, inputs):
            # Convert to contiguous float32 numpy array (move to CPU first for CUDA tensors)
            arr = tensor.detach().cpu().contiguous().float().numpy()
            data = arr.flatten().tolist()
            self._graph.set_input(name, data)

            # If this input has a _kn (transposed) variant, set that too
            if name in self._kn_inputs:
                kn_name = self._kn_inputs[name]
                # _kn tensors are 2D matrix transposes
                if arr.ndim == 2:
                    transposed = arr.T.flatten().tolist()
                    self._graph.set_input(kn_name, transposed)

        # Run the graph
        self._graph.run()

        # Get outputs and convert back to PyTorch tensors on the same device as inputs
        outputs = []
        for name, shape in zip(self._output_names, self._output_shapes):
            data = self._graph.get_output(name)
            tensor = torch.tensor(data, dtype=torch.float32).reshape(tuple(shape)).to(input_device)
            outputs.append(tensor)

        # Return as a tuple (TorchDynamo expects tuple return from backend callables)
        return tuple(outputs)
