"""CompiledModel wrapper for the Rust CompiledGraph."""

from typing import List, Optional

import torch


class CompiledModel:
    """Wrapper around CompiledGraph that handles PyTorch tensor conversion."""

    def __init__(self, graph_result, weight_refs=None, input_names=None, user_indices=None):
        """Initialize with a compiled CompiledGraph from Rust.

        Args:
            graph_result: The CompiledGraph from luminal_python.process_onnx() or process_pt2()
            weight_refs: List of PyTorch tensors to keep alive (prevents GC of shared weights)
            input_names: Override for user input names. If None, uses graph_result.input_names.
            user_indices: When torch.compile lifts model parameters into extra args,
                this tells __call__ which arg positions are actual user inputs.
                None means all args are user inputs (PT2 path).
        """
        self._graph = graph_result
        self._input_names = input_names or graph_result.input_names
        self._output_names = graph_result.output_names
        self._output_shapes = graph_result.output_shapes
        self._has_dynamic_dims = getattr(graph_result, 'has_dynamic_dims', False)
        self._weight_refs = weight_refs or []
        self._user_indices = user_indices
        self._is_cuda = (graph_result.backend == "cuda")

    def set_dim(self, param_name: str, value: int) -> None:
        """Set a dynamic dimension value by its param name."""
        self._graph.set_dim(param_name, value)

    @property
    def has_dynamic_dims(self) -> bool:
        return self._has_dynamic_dims

    @property
    def dim_params(self) -> List[str]:
        return self._graph.dim_params

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Execute the compiled model with PyTorch tensor inputs.

        Args:
            *inputs: PyTorch tensors. When torch.compile lifts model parameters,
                this includes both weights and user inputs. user_indices filters
                to just the user inputs.

        Returns:
            Tuple of PyTorch tensors containing the model outputs
        """
        # Extract user inputs (torch.compile may pass lifted weights as extra args)
        if self._user_indices is not None:
            user_inputs = [inputs[i] for i in self._user_indices]
        else:
            if len(inputs) != len(self._input_names):
                raise ValueError(
                    f"Expected {len(self._input_names)} inputs, got {len(inputs)}"
                )
            user_inputs = inputs

        input_device = inputs[0].device if inputs else torch.device("cpu")

        # Auto-detect dynamic dims from input shapes
        if self._has_dynamic_dims:
            input_shapes = [list(t.shape) for t in user_inputs]
            self._graph.auto_set_dims_from_input_shapes(input_shapes)

        # Set user input data via pointer (avoids Python list conversion)
        for name, tensor in zip(self._input_names, user_inputs):
            if self._is_cuda and tensor.is_cuda:
                t = tensor.detach().contiguous().float()
                self._graph.set_input_device_ptr(name, t.data_ptr(), t.numel() * 4)
            else:
                t = tensor.detach().cpu().contiguous().float()
                self._graph.set_input_from_ptr(name, t.data_ptr(), t.numel())

        # Run the graph
        self._graph.run()

        # Get output shapes — resolve dynamically if needed
        if self._has_dynamic_dims:
            output_shapes = self._graph.resolve_output_shapes()
        else:
            output_shapes = self._output_shapes

        # Get outputs and convert back to PyTorch tensors on the same device as inputs
        outputs = []
        for name, shape in zip(self._output_names, output_shapes):
            data = self._graph.get_output(name)
            tensor = (
                torch.tensor(data, dtype=torch.float32)
                .reshape(tuple(shape))
                .to(input_device)
            )
            outputs.append(tensor)

        return tuple(outputs)
