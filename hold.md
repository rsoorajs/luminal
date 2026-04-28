  Landed on branch:
  - 6e7ee558 — test suite (the spec)
  - f6845011 — scaffold (identity marker kernels)
  - 44324f1c — Binary→Unary pair-fuse rules (12 rules, got 15/18 green)

  In your working tree, uncommitted, compiles but untested:
  - 44 more rules added to FusionEnd::rewrites(): Unary→Binary (both positions), Binary→Binary (both positions), FusionEnd→Unary grow, FusionEnd→Binary grow (both positions), and the merge rule for the diamond's top Add.
  - Total: 56 rules.

  First thing to do tomorrow:
  cargo test -p luminal_cuda_lite --lib fusion::
  See if chain_of_binaries, unary_then_binary, and diamond_dag_fuses now pass. If diamond passes, also tighten the two invariant tests (they currently match cascade artifacts via len() == 5; should match the exact ["Add","Add","Exp","Mul","Sin"] op set).

  Sleep well 🌙
