use anyhow::Result;
use luminal::prelude::*;

use crate::ops_parse::convolution::{conv_unfold, depthwise_conv};
use crate::pt2_schema::*;

use super::Translator;

const CONV_INPUT_ARG: usize = 0;
const CONV_WEIGHT_ARG: usize = 1;
const CONV_BIAS_ARG: usize = 2;
const CONV_STRIDE_ARG: usize = 3;
const CONV_PADDING_ARG: usize = 4;
const CONV_DILATION_ARG: usize = 5;
const CONV_GROUPS_ARG: usize = 6;

const CONVOLUTION_TRANSPOSED_ARG: usize = 6;
const CONVOLUTION_OUTPUT_PADDING_ARG: usize = 7;
const CONVOLUTION_GROUPS_ARG: usize = 8;

impl<'a> Translator<'a> {
    /// Translate aten.conv{1,2,3}d.default and aten.convolution.default.
    ///
    /// The PT2 export may omit defaulted trailing arguments entirely. In practice this means
    /// conv{N}d.default can show up as just `(input, weight)` for the no-bias, stride=1,
    /// padding=0, dilation=1, groups=1 case.
    pub(crate) fn translate_conv(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, CONV_INPUT_ARG)?;
        let weight = self.get_input_tensor(node, CONV_WEIGHT_ARG)?;
        let bias = self.get_input_tensor(node, CONV_BIAS_ARG).ok();

        let x_dims = input.dims();
        let w_dims = weight.dims();
        let rank = x_dims.len();
        let spatial = rank - 2;

        let stride = self
            .get_ints_arg(node, CONV_STRIDE_ARG)
            .unwrap_or_else(|_| vec![1; spatial]);
        let padding = self
            .get_ints_arg(node, CONV_PADDING_ARG)
            .unwrap_or_else(|_| vec![0; spatial]);
        let mut dilation = self
            .get_ints_arg(node, CONV_DILATION_ARG)
            .unwrap_or_else(|_| vec![1; spatial]);
        let groups = if node.target == "torch.ops.aten.convolution.default" {
            let transposed = self
                .get_bool_arg(node, CONVOLUTION_TRANSPOSED_ARG)
                .unwrap_or(false);
            anyhow::ensure!(
                !transposed,
                "conv: ConvTranspose / transposed=true is not supported yet"
            );
            let output_padding = self
                .get_ints_arg(node, CONVOLUTION_OUTPUT_PADDING_ARG)
                .unwrap_or_else(|_| vec![0; spatial]);
            anyhow::ensure!(
                output_padding.iter().all(|&v| v == 0),
                "conv: output_padding is not supported for non-transposed convolution"
            );
            self.get_int_arg(node, CONVOLUTION_GROUPS_ARG).unwrap_or(1) as usize
        } else {
            self.get_int_arg(node, CONV_GROUPS_ARG).unwrap_or(1) as usize
        };
        if dilation.len() != spatial {
            dilation = vec![1; spatial];
        }

        let ch_out = w_dims[0]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("conv: weight C_out must be concrete"))?;
        let ch_in = x_dims[1]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("conv: input C_in must be concrete"))?;
        anyhow::ensure!(
            stride.len() == spatial && padding.len() == spatial && dilation.len() == spatial,
            "conv: stride/padding/dilation rank must match spatial rank {spatial}"
        );
        anyhow::ensure!(
            groups > 0 && ch_in % groups == 0 && ch_out % groups == 0,
            "conv: invalid group configuration (C_in={ch_in}, C_out={ch_out}, groups={groups})"
        );
        let ch_per_group = ch_in / groups;

        let kernel_shape: Vec<usize> = w_dims[2..]
            .iter()
            .map(|d| {
                d.to_usize()
                    .ok_or_else(|| anyhow::anyhow!("conv: kernel dims must be concrete"))
            })
            .collect::<Result<_>>()?;
        let kernel_product: usize = kernel_shape.iter().product();

        // ATen uses symmetric padding (same begin/end)
        let stride_u: Vec<usize> = stride.iter().map(|&v| v as usize).collect();
        let padding_u: Vec<usize> = padding.iter().map(|&v| v as usize).collect();
        let dilation_u: Vec<usize> = dilation.iter().map(|&v| v as usize).collect();

        let mut out = if groups > 1 {
            let group_out = ch_out / groups;

            if ch_per_group == 1 {
                // Depthwise (including channel multiplier > 1): avoid per-channel slicing.
                depthwise_conv(
                    input,
                    weight,
                    &kernel_shape,
                    &stride_u,
                    &dilation_u,
                    &padding_u,
                    &padding_u,
                    ch_in,
                    group_out,
                    kernel_product,
                    spatial,
                )
            } else {
                // General grouped: pre-pad full input then slice per group
                let padded_input = {
                    let mut pad_spec: Vec<(Expression, Expression)> =
                        vec![(0.into(), 0.into()); 2 + spatial];
                    for i in 0..spatial {
                        pad_spec[2 + i] = (padding_u[i].into(), padding_u[i].into());
                    }
                    input.pad(pad_spec, 0.0)
                };

                let no_pad = vec![0usize; spatial];
                let mut group_outputs = Vec::with_capacity(groups);
                for g in 0..groups {
                    let x_g = slice_channel_group(padded_input, g, ch_per_group, spatial);
                    let w_g =
                        slice_weight_group(weight, g, group_out, ch_per_group * kernel_product);
                    group_outputs.push(conv_unfold(
                        x_g,
                        w_g,
                        &kernel_shape,
                        &stride_u,
                        &dilation_u,
                        &no_pad,
                        &no_pad,
                        ch_per_group,
                        group_out,
                        spatial,
                    ));
                }

                let mut result = group_outputs[0];
                for g_out in &group_outputs[1..] {
                    result = result.concat_along(*g_out, 1);
                }
                result
            }
        } else {
            let mut w_flat = weight;
            w_flat.shape = ShapeTracker::new_with_element_bits(
                vec![ch_out, ch_in * kernel_product],
                weight.dtype.bits(),
            );

            conv_unfold(
                input,
                w_flat,
                &kernel_shape,
                &stride_u,
                &dilation_u,
                &padding_u,
                &padding_u,
                ch_in,
                ch_out,
                spatial,
            )
        };

        if let Some(b) = bias {
            let out_dims = out.dims();
            let mut b_expanded = b.expand_dim(0, 1);
            for i in 0..spatial {
                b_expanded = b_expanded.expand_dim(2 + i, out_dims[2 + i]);
            }
            out += b_expanded;
        }

        Ok(out)
    }
}

/// Slice input channels for one group.
/// Caller must pre-pad `x` so no additional padding is applied to the slice.
fn slice_channel_group(
    x: GraphTensor,
    g: usize,
    ch_per_group: usize,
    spatial: usize,
) -> GraphTensor {
    let start = g * ch_per_group;
    let end = start + ch_per_group;
    let dims = x.dims();
    let rank = 2 + spatial;
    let mut slices: Vec<(Expression, Expression)> = Vec::with_capacity(rank);
    slices.push((0.into(), dims[0]));
    slices.push((start.into(), end.into()));
    for dim in dims.iter().take(rank).skip(2) {
        slices.push((0.into(), *dim));
    }
    x.slice(slices)
}

/// Slice and flatten weight for one group.
fn slice_weight_group(
    w: GraphTensor,
    g: usize,
    group_out: usize,
    flat_inner: usize,
) -> GraphTensor {
    let start = g * group_out;
    let end = start + group_out;
    let w_dims = w.dims();
    let mut slices: Vec<(Expression, Expression)> = Vec::with_capacity(w_dims.len());
    slices.push((start.into(), end.into()));
    for dim in w_dims.iter().skip(1) {
        slices.push((0.into(), *dim));
    }
    // Materialize through Add: binary op outputs are contiguous in Luminal, which makes the
    // following flatten safe for the sliced weight buffer.
    let w_sliced = w.slice(slices) + 0.0;
    let mut w_flat = w_sliced;
    w_flat.shape =
        ShapeTracker::new_with_element_bits(vec![group_out, flat_inner], w_sliced.dtype.bits());
    w_flat
}
