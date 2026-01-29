#![allow(unused)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::*;
use tracing_perfetto_sdk_schema::{self as schema, TrackEvent, debug_annotation::NameField, trace_packet, track_event};
use uuid::Uuid;

pub mod cuda_graph;
pub mod ops;

pub use cuda_graph::*;
pub use ops::Ops;

/// Record CUDA graph kernel timings as nested slices in perfetto trace
pub fn record_cuda_graph_timings(
    trace: &schema::Trace,
    cuda_graph_timings: &[(CudaGraphTiming, Uuid)],
) -> Vec<schema::TracePacket> {
    let mut packets = Vec::new();
    for (graph_timing, id) in cuda_graph_timings {
        let parent_info = trace.packet.iter().find_map(|p| {
            match &p.data {
                Some(trace_packet::Data::TrackEvent(TrackEvent {
                    r#type: ty, track_uuid, debug_annotations, ..
                })) if *ty == Some(track_event::Type::SliceBegin as i32)
                    && debug_annotations.iter().any(|a| {
                        matches!((&a.name_field, &a.value),
                            (Some(NameField::Name(k)), Some(tracing_perfetto_sdk_schema::debug_annotation::Value::StringValue(v)))
                            if k == "id" && *v == format!("{id}"))
                    }) =>
                {
                    Some((p.timestamp?, p.timestamp_clock_id?, (*track_uuid)?,
                        match &p.optional_trusted_packet_sequence_id {
                            Some(trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(seq)) => *seq,
                            _ => 0,
                        }))
                }
                _ => None,
            }
        });
        let Some((host_time, clock_id, track_uuid, sequence_id)) = parent_info else { continue };
        let launch_offset = graph_timing.launch_latency_ns;
        for kernel_timing in &graph_timing.kernel_timings {
            packets.push(schema::TracePacket {
                timestamp: Some(host_time + launch_offset + kernel_timing.start_ns),
                timestamp_clock_id: Some(clock_id),
                optional_trusted_packet_sequence_id: Some(trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(sequence_id)),
                data: Some(trace_packet::Data::TrackEvent(schema::TrackEvent {
                    track_uuid: Some(track_uuid),
                    r#type: Some(track_event::Type::SliceBegin as i32),
                    name_field: Some(track_event::NameField::Name(kernel_timing.kernel_name.to_owned())),
                    ..Default::default()
                })),
                ..Default::default()
            });
            packets.push(schema::TracePacket {
                timestamp: Some(host_time + launch_offset + kernel_timing.end_ns),
                timestamp_clock_id: Some(clock_id),
                optional_trusted_packet_sequence_id: Some(trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(sequence_id)),
                data: Some(trace_packet::Data::TrackEvent(schema::TrackEvent {
                    track_uuid: Some(track_uuid),
                    r#type: Some(track_event::Type::SliceEnd as i32),
                    name_field: Some(track_event::NameField::Name(kernel_timing.kernel_name.to_owned())),
                    ..Default::default()
                })),
                ..Default::default()
            });
        }
    }
    packets
}

pub trait KernelOp: luminal::op::EgglogOp {
    #[allow(clippy::type_complexity)]
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    );

    /// Returns the output buffer size in elements.
    fn output_size(&self) -> Expression;

    /// Returns the number of bytes this kernel will load from global memory.
    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    /// Returns the number of bytes this kernel will store to global memory.
    fn bytes_stored(&self) -> Expression {
        0.into()
    }

    /// Returns the number of floating point operations this kernel performs.
    fn flops(&self) -> Expression {
        0.into()
    }

    /// Returns the name of this kernel for profiling display.
    fn kernel_name(&self) -> &'static str {
        "Unknown"
    }
}

luminal::impl_into_ops!(KernelOp);
