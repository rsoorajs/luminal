use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

impl<'a> Translator<'a> {
    pub(crate) fn translate_arange(&mut self, node: &Node) -> Result<GraphTensor> {
        let positional_args: Vec<Expression> = node.inputs.iter()
            .filter(|i| i.kind <= 1)
            .filter_map(|i| self.resolve_arg_as_expression(&i.arg))
            .collect();

        match positional_args.len() {
            0 => anyhow::bail!("arange: no positional args found"),
            1 => Ok(self.graph.arange(positional_args[0])),
            _ => Ok(self.graph.arange_options(
                positional_args[0],
                positional_args[1],
                1,
            )),
        }
    }

    pub(crate) fn translate_full(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.get_exprs_arg(node, 0)?;
        let val = self.get_float_arg(node, 1)? as f32;
        Ok(self.graph.constant_float(val).expand_rhs(shape))
    }

    pub(crate) fn translate_zeros(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_name = node.outputs.first()
            .and_then(|o| o.as_tensor.as_ref())
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let meta = self.tensor_meta(&output_name)
            .context("Missing tensor meta for zeros output")?;
        let shape = self.tensor_meta_to_shape(meta)?;
        Ok(self.graph.constant_float(0.0).expand_rhs(shape))
    }

    pub(crate) fn translate_ones(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_name = node.outputs.first()
            .and_then(|o| o.as_tensor.as_ref())
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let meta = self.tensor_meta(&output_name)
            .context("Missing tensor meta for ones output")?;
        let shape = self.tensor_meta_to_shape(meta)?;
        Ok(self.graph.constant_float(1.0).expand_rhs(shape))
    }

    pub(crate) fn translate_new_ones(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_name = node.outputs.first()
            .and_then(|o| o.as_tensor.as_ref())
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let meta = self.tensor_meta(&output_name)
            .context("Missing tensor meta for new_ones output")?;
        let shape = self.tensor_meta_to_shape(meta)?;
        if shape.is_empty() {
            Ok(self.graph.constant_float(1.0))
        } else {
            Ok(self.graph.constant_float(1.0).expand_rhs(shape))
        }
    }

    pub(crate) fn translate_where(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let y = self.get_input_tensor(node, 2)?;
        let cond_f32 = cond.cast(DType::F32);
        let (cond_f32_x, x_b) = broadcast_binary(cond_f32, x);
        let one = self.graph.constant_float(1.0).expand_rhs(cond_f32.shape);
        let inv_cond = one - cond_f32;
        let (inv_cond_y, y_b) = broadcast_binary(inv_cond, y);
        Ok(cond_f32_x * x_b + inv_cond_y * y_b)
    }

    pub(crate) fn translate_where_scalar_other(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let other_val = self.get_float_arg(node, 2)? as f32;
        let cond_f32 = cond.cast(DType::F32);
        let (c, x_b) = broadcast_binary(cond_f32, x);
        let one = self.graph.constant_float(1.0).expand_rhs(cond_f32.shape);
        let inv = one - cond_f32;
        let other = self.graph.constant_float(other_val).expand_rhs(inv.shape);
        Ok(c * x_b + inv * other)
    }

    pub(crate) fn translate_diff(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(-1)
        } else {
            -1
        };
        let dim = normalize_dim(dim, input.shape.len());

        let prepend = if node.inputs.len() > 3 {
            self.get_input_tensor(node, 3).ok()
        } else {
            None
        };

        let x = if let Some(prep) = prepend {
            prep.concat_along(input, dim)
        } else {
            input
        };

        let dim_size = x.shape.dims[dim];
        let front = x.slice_along(Expression::from(1)..dim_size, dim);
        let back = x.slice_along(Expression::from(0)..dim_size - 1, dim);
        Ok(front - back)
    }

    pub(crate) fn translate_tril(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let diagonal = if node.inputs.len() > 1 {
            self.get_int_arg(node, 1).unwrap_or(0) as i32
        } else {
            0
        };
        let dims = a.shape.dims.clone();
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        // tril mask: row >= col - diagonal
        let size = if let (Some(r), Some(c)) = (rows.to_usize(), cols.to_usize()) {
            r.max(c)
        } else {
            // fallback
            return Ok(a);
        };
        let mask = self.graph.tril(size, diagonal).cast(DType::F32);
        // Slice mask to [rows, cols] if not square
        let mask = if rows != cols {
            let r = rows.to_usize().unwrap();
            let c = cols.to_usize().unwrap();
            mask.slice_along(0..r, 0).slice_along(0..c, 1)
        } else {
            mask
        };
        // Broadcast mask to match batch dims
        let mut mask_expanded = mask;
        for i in (0..dims.len() - 2).rev() {
            mask_expanded = mask_expanded.expand_dim(0, dims[i]);
        }
        Ok(a * mask_expanded)
    }

    pub(crate) fn translate_triu(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let diagonal = if node.inputs.len() > 1 {
            self.get_int_arg(node, 1).unwrap_or(0) as i32
        } else {
            0
        };
        let dims = a.shape.dims.clone();
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        let size = if let (Some(r), Some(c)) = (rows.to_usize(), cols.to_usize()) {
            r.max(c)
        } else {
            return Ok(a);
        };
        let mask = self.graph.triu(size, diagonal).cast(DType::F32);
        let mask = if rows != cols {
            let r = rows.to_usize().unwrap();
            let c = cols.to_usize().unwrap();
            mask.slice_along(0..r, 0).slice_along(0..c, 1)
        } else {
            mask
        };
        let mut mask_expanded = mask;
        for i in (0..dims.len() - 2).rev() {
            mask_expanded = mask_expanded.expand_dim(0, dims[i]);
        }
        Ok(a * mask_expanded)
    }

    pub(crate) fn translate_topk(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let k = self.get_int_arg(node, 1)? as usize;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(-1)
        } else {
            -1
        };
        let dim = normalize_dim(dim, a.shape.len());

        let values = a.topk_values(k, dim);
        let indices = a.topk_indexes(k, dim);

        // Store second output (indices) — check as_tensors (multi-output) or as_tensor
        if let Some(ts) = node.outputs.first().and_then(|o| o.as_tensors.as_ref()) {
            if ts.len() > 1 {
                self.tensors.insert(ts[1].name.clone(), indices);
            }
        } else if node.outputs.len() > 1 {
            if let Some(idx_name) = node.outputs[1].as_tensor.as_ref() {
                self.tensors.insert(idx_name.name.clone(), indices);
            }
        }

        Ok(values)
    }

    pub(crate) fn translate_one_hot(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let num_classes = self.get_int_arg(node, 1)? as usize;
        // one_hot: output[..., i] = 1 if input[...] == i else 0
        let a_int = a.cast(DType::Int);
        let classes = self.graph.arange(num_classes);
        // Expand a to [..., 1] and classes to [..., num_classes]
        let a_expanded = a_int.expand_dim(a.shape.len(), num_classes);
        let mut classes_expanded = classes;
        for d in a.shape.dims.iter().rev() {
            classes_expanded = classes_expanded.expand_dim(0, *d);
        }
        Ok(a_expanded.eq(classes_expanded).cast(DType::Int))
    }

    pub(crate) fn translate_wrap_set_grad(&mut self, node: &Node) -> Result<()> {
        let subgraph = node.inputs[1].arg.as_graph()
            .context("wrap_with_set_grad: missing subgraph")?
            .clone();

        let sg_inputs = &subgraph.graph.inputs;
        let forwarded_args = &node.inputs[2..];
        for (sg_input, fwd_arg) in sg_inputs.iter().zip(forwarded_args) {
            if let Some(sg_name) = sg_input.as_tensor.as_ref() {
                if let Some(main_name) = fwd_arg.arg.as_tensor_name() {
                    let tensor = self.get_tensor(main_name)?;
                    self.tensors.insert(sg_name.name.clone(), tensor);
                }
            }
        }

        for (k, v) in &subgraph.graph.tensor_values {
            self.extra_tensor_values.insert(k.clone(), v.clone());
        }

        let sg_nodes = subgraph.graph.nodes.clone();
        for (i, sg_node) in sg_nodes.iter().enumerate() {
            self.translate_node(sg_node)
                .with_context(|| format!("Subgraph node {i}: {}", sg_node.target))?;
        }

        for (main_out, sg_out) in node.outputs.iter().zip(subgraph.graph.outputs.iter()) {
            if let (Some(main_name), Some(sg_name)) = (
                main_out.as_tensor.as_ref(),
                sg_out.as_tensor.as_ref(),
            ) {
                if main_name.name != sg_name.name {
                    let tensor = self.get_tensor(&sg_name.name)?;
                    self.tensors.insert(main_name.name.clone(), tensor);
                }
            }
        }

        Ok(())
    }
}
