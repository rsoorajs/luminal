#!/usr/bin/env bash
# Generate PNGs for each fusion-rule LHS/RHS tree.
set -euo pipefail
cd "$(dirname "$0")"

# Common dot prelude: top-to-bottom, rounded filled boxes, Helvetica.
PRELUDE='rankdir=TB; nodesep=0.35; ranksep=0.4;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14, penwidth=1.2];
  edge [arrowsize=0.7, color="#444"];'

# Color palette (semantic):
#   FE      -> red    (#f8d7da)
#   FS      -> green  (#d1e7dd)
#   Unary U -> blue   (#cfe2ff)
#   Binary B-> purple (#e0cffc)
#   Extern  -> grey   (#f8f9fa)
#   Inner ref -> yellow (#fff3cd)

render() {
  local name="$1" body="$2"
  cat > "${name}.dot" <<EOF
digraph ${name} {
  ${PRELUDE}
${body}
}
EOF
  dot -Tpng -Gdpi=150 "${name}.dot" -o "${name}.png"
}

# -- Rule 1: Seed Unary -----------------------------------------------------
render rule1_lhs '
  U [label="U", fillcolor="#cfe2ff"];
  x [label="x", fillcolor="#f8f9fa"];
  U -> x;'

render rule1_rhs '
  FE [label="FE", fillcolor="#f8d7da"];
  U  [label="U",  fillcolor="#cfe2ff"];
  FS [label="FS", fillcolor="#d1e7dd"];
  x  [label="x",  fillcolor="#f8f9fa"];
  FE -> U -> FS -> x;'

# -- Rule 2: Seed Binary ----------------------------------------------------
render rule2_lhs '
  B [label="B", fillcolor="#e0cffc"];
  a [label="a", fillcolor="#f8f9fa"];
  b [label="b", fillcolor="#f8f9fa"];
  B -> a; B -> b;'

render rule2_rhs '
  FE [label="FE", fillcolor="#f8d7da"];
  B  [label="B",  fillcolor="#e0cffc"];
  FSa [label="FS", fillcolor="#d1e7dd"];
  FSb [label="FS", fillcolor="#d1e7dd"];
  a [label="a", fillcolor="#f8f9fa"];
  b [label="b", fillcolor="#f8f9fa"];
  FE -> B; B -> FSa; B -> FSb; FSa -> a; FSb -> b;'

# -- Rule 3: Grow FE -> Unary ----------------------------------------------
render rule3_lhs '
  U     [label="U",     fillcolor="#cfe2ff"];
  FE    [label="FE",    fillcolor="#f8d7da"];
  inner [label="inner", fillcolor="#fff3cd"];
  U -> FE -> inner;'

render rule3_rhs '
  FE    [label="FE",    fillcolor="#f8d7da"];
  U     [label="U",     fillcolor="#cfe2ff"];
  inner [label="inner", fillcolor="#fff3cd"];
  FE -> U -> inner;'

# -- Rule 4: Grow FE -> Binary, FE on LHS -----------------------------------
render rule4_lhs '
  B       [label="B",       fillcolor="#e0cffc"];
  FE      [label="FE",      fillcolor="#f8d7da"];
  inner_a [label="inner_a", fillcolor="#fff3cd"];
  b       [label="b\n(external)", fillcolor="#f8f9fa"];
  B -> FE; B -> b; FE -> inner_a;'

render rule4_rhs '
  FE_new  [label="FE",      fillcolor="#f8d7da"];
  B       [label="B",       fillcolor="#e0cffc"];
  inner_a [label="inner_a", fillcolor="#fff3cd"];
  FS      [label="FS\n(NEW)", fillcolor="#d1e7dd"];
  b       [label="b",       fillcolor="#f8f9fa"];
  FE_new -> B; B -> inner_a; B -> FS; FS -> b;'

# -- Rule 5: Grow FE -> Binary, FE on RHS (mirror of 4) ---------------------
render rule5_lhs '
  B       [label="B",       fillcolor="#e0cffc"];
  a       [label="a\n(external)", fillcolor="#f8f9fa"];
  FE      [label="FE",      fillcolor="#f8d7da"];
  inner_b [label="inner_b", fillcolor="#fff3cd"];
  B -> a; B -> FE; FE -> inner_b;'

render rule5_rhs '
  FE_new  [label="FE",      fillcolor="#f8d7da"];
  B       [label="B",       fillcolor="#e0cffc"];
  FS      [label="FS\n(NEW)", fillcolor="#d1e7dd"];
  a       [label="a",       fillcolor="#f8f9fa"];
  inner_b [label="inner_b", fillcolor="#fff3cd"];
  FE_new -> B; B -> FS; B -> inner_b; FS -> a;'

# -- Rule 6: Merge ----------------------------------------------------------
render rule6_lhs '
  B       [label="B",       fillcolor="#e0cffc"];
  FEa     [label="FE",      fillcolor="#f8d7da"];
  FEb     [label="FE",      fillcolor="#f8d7da"];
  inner_a [label="inner_a", fillcolor="#fff3cd"];
  inner_b [label="inner_b", fillcolor="#fff3cd"];
  B -> FEa; B -> FEb; FEa -> inner_a; FEb -> inner_b;'

render rule6_rhs '
  FE      [label="FE",      fillcolor="#f8d7da"];
  B       [label="B",       fillcolor="#e0cffc"];
  inner_a [label="inner_a", fillcolor="#fff3cd"];
  inner_b [label="inner_b", fillcolor="#fff3cd"];
  FE -> B; B -> inner_a; B -> inner_b;'

# -- Cascade illustration (Rule 1 self-rematch) -----------------------------
render cascade '
  FE1 [label="FE",   fillcolor="#f8d7da"];
  U1  [label="U",    fillcolor="#cfe2ff"];
  FS1 [label="FS",   fillcolor="#d1e7dd"];
  FS2 [label="FS",   fillcolor="#d1e7dd"];
  x   [label="x",    fillcolor="#f8f9fa"];
  FE1 -> U1 -> FS1 -> FS2 -> x;
  label="After Rule 1 re-fires on its own output:\nthe inner U(FS(x)) gets unioned with FE(U(FS(FS(x))))";
  labelloc=b;
  fontname="Helvetica"; fontsize=12;'

# -- Diamond DAG: final fused form ------------------------------------------
render diamond '
  FE      [label="FE",                fillcolor="#f8d7da"];
  outAdd  [label="Add\n(out = w + v)", fillcolor="#e0cffc"];
  Mul     [label="Mul\n(w = u * a)",   fillcolor="#e0cffc"];
  Sin     [label="Sin\n(v)",           fillcolor="#cfe2ff"];
  Exp2    [label="Exp2\n(u)",          fillcolor="#cfe2ff"];
  innerAdd[label="Add\n(t = a + b)",   fillcolor="#e0cffc"];
  FSa1    [label="FS_a1",              fillcolor="#d1e7dd"];
  FSa2    [label="FS_a2\n(spurious)",  fillcolor="#d1e7dd", color="#dc3545", penwidth=2];
  FSb1    [label="FS_b1",              fillcolor="#d1e7dd"];
  a       [label="a",                  fillcolor="#f8f9fa"];
  b       [label="b",                  fillcolor="#f8f9fa"];

  FE       -> outAdd;
  outAdd   -> Mul;
  outAdd   -> Sin;
  Mul      -> Exp2;
  Mul      -> FSa2;
  Sin      -> innerAdd;
  Exp2     -> innerAdd;
  innerAdd -> FSa1;
  innerAdd -> FSb1;
  FSa1     -> a;
  FSa2     -> a;
  FSb1     -> b;

  label="Final fused diamond: 3 FS nodes for 2 distinct external tensors {a, b}.\nFS_a1 and FS_a2 both wrap a but are never unified.";
  labelloc=b;
  fontname="Helvetica"; fontsize=12;'

echo "rendered $(ls *.png | wc -l) PNGs"
