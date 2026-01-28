# `luminal_bench`

Luminal 的基准测试与调试工具箱（Criterion benchmarks + egglog 调试工具）。

## 运行基准测试

目前 crate 的 bench 默认以 Metal 为例（通过 feature 开关启用）。

```bash
# L1: micro（单算子/HLIR primitive）
cargo bench -p luminal_bench --features metal --bench micro

# L2: patterns（组合算子/模式）
cargo bench -p luminal_bench --features metal --bench patterns
```

### 产物位置（Criterion）

运行后常见输出位于：

- HTML 报告：`target/criterion/report/index.html`
- micro 的 metrics 映射：`target/criterion/bench_metrics.json`
- micro 的完整汇总：`target/criterion/bench_report.json`
- patterns 的 metrics 映射：`target/criterion/pattern_metrics.json`
- patterns 的完整汇总：`target/criterion/pattern_report.json`

这些 JSON（bytes/flops 等常量指标）结合 Criterion 的时间结果，可以计算吞吐、MBU、MFU 等派生指标。

## 基准覆盖范围（概览）

### L1 micro（单算子）

覆盖 HLIR primitives 的单算子性能（当前实现包含）：

- Unary：`Exp2` / `Log2` / `Sin` / `Recip` / `Sqrt`
- Binary：`Add` / `Mul` / `Mod` / `LessThan`
- Indexing：`Gather` / `Cast`
- Reduction：`Sum` / `Max`

### L2 patterns（组合模式）

覆盖常见组合模式（当前实现包含）：

- `MatMul`
- `Softmax`
- `GeLU`
- `Attention`
- `LayerNorm`（目前在 metal bench 中会跳过：需要尚未支持的 HLIR primitives）

## egglog 调试工具：`debug_ops`

`examples/debug_ops.rs` 是一个通用的 egglog/降级调试工具，用来定位：

- 为什么某个 HLIR op 没有被后端 op 命中（cleanup 后导致 `No valid graphs present...`）
- 为什么某个 function（如 `dtype`）在某些节点上缺失

### 常用命令（Metal 示例）

```bash
# 默认：只打印摘要（HLIR/egglog op 统计 + root），并运行 build_search_space（会打印 rule matches）
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner

# 显式做 op 覆盖检查：指定 HLIR:Backend 映射
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --inspect-op Add:MetalAdd

# 运行更完整的 lowering 分析输出
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --analyze --inspect-op Add:MetalAdd

# 追踪“第一个缺失 dtype 的 Add”（HLIR-only）
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --trace-missing-dtype

# 追踪任意 egglog function 的链路（HLIR-only）
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --trace-fn dtype t24

# 查看某个变量的 eclass/enodes/children/dtype 事实（HLIR-only）
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --inspect-var t24

# 导出结构化 JSON（便于复现与对比）
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --json target/debug_ops.json
```

更多参数见：

```bash
cargo run -p luminal_bench --features metal --example debug_ops -- --help
```
