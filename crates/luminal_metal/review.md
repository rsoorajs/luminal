Metal ops.rs review (updated)

主要问题
- 高: MetalGather 丢失 data_shape, 并用 out_shape 计算 data 索引, 当 data_shape != index_shape 时会读错. crates/luminal_metal/src/kernel/ops.rs:1166 crates/luminal_metal/src/kernel/ops.rs:1150 crates/luminal_metal/src/kernel/ops.rs:1205
- 高: dtype 传播与 kernel 实现不一致. rewrite 直接继承输入 dtype, 但 kernel 全部按 float 实现, 对 Int/F16/Bf16 会静默错误. crates/luminal_metal/src/kernel/ops.rs:54 crates/luminal_metal/src/kernel/ops.rs:432 crates/luminal_metal/src/kernel/ops.rs:549 crates/luminal_metal/src/kernel/ops.rs:671 crates/luminal_metal/src/kernel/ops.rs:1220
- 中: 多数 output_size() 强制 max(1), 空张量会被当作 1 元素执行, 可能越界或语义错误. crates/luminal_metal/src/kernel/ops.rs:137 crates/luminal_metal/src/kernel/ops.rs:263 crates/luminal_metal/src/kernel/ops.rs:380 crates/luminal_metal/src/kernel/ops.rs:498 crates/luminal_metal/src/kernel/ops.rs:616 crates/luminal_metal/src/kernel/ops.rs:773 crates/luminal_metal/src/kernel/ops.rs:927 crates/luminal_metal/src/kernel/ops.rs:1237
- 低: MetalIota 在 range == 0 时 dispatch 0 threadgroups, 可能触发 Metal 运行时错误. crates/luminal_metal/src/kernel/ops.rs:1120 crates/luminal_metal/src/kernel/ops.rs:1132
- 低: MetalConstant 直接把 f32 文本插入 shader, NaN/Inf 可能生成非法 Metal 代码. crates/luminal_metal/src/kernel/ops.rs:972

疑问/假设
- Metal 后端是否只打算支持 F32? 如果是, 建议在 rewrite 阶段限制 dtype 或在 load_llir 阶段显式拒绝非 F32.
- Gather 是否保证 index_shape == data_shape? 如果不保证, 需要保存 data_shape 并用它计算 data_idx.

测试缺口
- 缺少针对非 F32 (Int/F16/Bf16), Gather 的 data_shape != index_shape, 以及零尺寸张量的用例.
