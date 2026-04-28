#!/usr/bin/env bash
# Generate PNGs for the NEW (discriminator-based) fusion design.
set -euo pipefail
cd "$(dirname "$0")"

PRELUDE='rankdir=BT; nodesep=0.35; ranksep=0.4;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.2];
  edge [arrowsize=0.7];'

# Color palette:
#   FE       -> red    (#f8d7da)
#   FS       -> green  (#d1e7dd)
#   Op false -> blue   (#cfe2ff)   (op outside any region)
#   Op true  -> purple (#e0cffc)   (op inside a fused region)
#   External -> grey   (#f8f9fa)
#   Inner    -> yellow (#fff3cd)   (e-class ref to upstream FE body)
#   Eclass cluster -> light yellow (#fffaeb)

render() {
  local name="$1" body="$2"
  cat > "${name}.dot" <<EOF
digraph ${name} {
  ${PRELUDE}
${body}
}
EOF
  dot -Tpng -Gdpi=150 "${name}.dot" -o "${name}.png" 2>&1 | grep -v "is not a known color" || true
}

# -- 1. Discriminator concept ------------------------------------------------
render v2_discriminator '
  // LHS column
  subgraph cluster_lhs {
    label="LHS pattern (must match in egraph)"; style="dashed,filled"; fillcolor="#fffaeb";
    fontname="Helvetica"; fontsize=11;
    LU2 [label="U2 (false)", fillcolor="#cfe2ff"];
    LU1 [label="U1 (false)", fillcolor="#cfe2ff"];
    Lx  [label="?x",         fillcolor="#f8f9fa"];
    Lx -> LU1; LU1 -> LU2;
  }

  // RHS column
  subgraph cluster_rhs {
    label="RHS (built and unioned with U2 e-class)"; style="dashed,filled"; fillcolor="#fffaeb";
    fontname="Helvetica"; fontsize=11;
    RFE  [label="FE",          fillcolor="#f8d7da"];
    RU2  [label="U2 (true)",   fillcolor="#e0cffc"];
    RU1  [label="U1 (true)",   fillcolor="#e0cffc"];
    RFS  [label="FS",          fillcolor="#d1e7dd"];
    Rx   [label="?x",          fillcolor="#f8f9fa"];
    Rx -> RFS; RFS -> RU1; RU1 -> RU2; RU2 -> RFE;
  }

  // Connect with a labeled invisible edge so they sit side-by-side.
  LU2 -> RFE [style=invis];

  labelloc=b;
  fontname="Helvetica"; fontsize=11;
  label="The discriminator field (false / true) makes LHS and RHS structurally non-unifiable.\nIter 2 cannot re-bind ?x to the freshly-minted U1(true) — pattern requires false.\nResult: NO cascade, regardless of repeat count.";
'

# -- 2. Base case 1: U -> U --------------------------------------------------
render v2_base1_uu '
  subgraph cluster_lhs {
    label="LHS"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    a_U2 [label="U2 (false)", fillcolor="#cfe2ff"];
    a_U1 [label="U1 (false)", fillcolor="#cfe2ff"];
    a_x  [label="?x",         fillcolor="#f8f9fa"];
    a_x -> a_U1 -> a_U2;
  }
  subgraph cluster_rhs {
    label="RHS"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    b_FE  [label="FE",        fillcolor="#f8d7da"];
    b_U2  [label="U2 (true)", fillcolor="#e0cffc"];
    b_U1  [label="U1 (true)", fillcolor="#e0cffc"];
    b_FS  [label="FS",        fillcolor="#d1e7dd"];
    b_x   [label="?x",        fillcolor="#f8f9fa"];
    b_x -> b_FS -> b_U1 -> b_U2 -> b_FE;
  }
  a_U2 -> b_FE [style=invis];
'

# -- 3. Base case 2: B -> U --------------------------------------------------
render v2_base2_bu '
  subgraph cluster_lhs {
    label="LHS"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    a_U  [label="U (false)", fillcolor="#cfe2ff"];
    a_B  [label="B (false)", fillcolor="#cfe2ff"];
    a_a  [label="?a",        fillcolor="#f8f9fa"];
    a_b  [label="?b",        fillcolor="#f8f9fa"];
    a_a -> a_B; a_b -> a_B; a_B -> a_U;
  }
  subgraph cluster_rhs {
    label="RHS"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    b_FE  [label="FE",        fillcolor="#f8d7da"];
    b_U   [label="U (true)",  fillcolor="#e0cffc"];
    b_B   [label="B (true)",  fillcolor="#e0cffc"];
    b_FSa [label="FS",        fillcolor="#d1e7dd"];
    b_FSb [label="FS",        fillcolor="#d1e7dd"];
    b_a   [label="?a",        fillcolor="#f8f9fa"];
    b_b   [label="?b",        fillcolor="#f8f9fa"];
    b_a -> b_FSa; b_b -> b_FSb; b_FSa -> b_B; b_FSb -> b_B;
    b_B -> b_U -> b_FE;
  }
  a_U -> b_FE [style=invis];
'

# -- 4. Base case 3: U -> B (LHS variant) ------------------------------------
render v2_base3_ub '
  subgraph cluster_lhs {
    label="LHS (LHS variant; mirror exists for RHS)"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    a_B  [label="B (false)", fillcolor="#cfe2ff"];
    a_U  [label="U (false)", fillcolor="#cfe2ff"];
    a_a  [label="?a",        fillcolor="#f8f9fa"];
    a_b  [label="?b",        fillcolor="#f8f9fa"];
    a_a -> a_U; a_U -> a_B; a_b -> a_B;
  }
  subgraph cluster_rhs {
    label="RHS"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    b_FE  [label="FE",        fillcolor="#f8d7da"];
    b_B   [label="B (true)",  fillcolor="#e0cffc"];
    b_U   [label="U (true)",  fillcolor="#e0cffc"];
    b_FSa [label="FS",        fillcolor="#d1e7dd"];
    b_FSb [label="FS",        fillcolor="#d1e7dd"];
    b_a   [label="?a",        fillcolor="#f8f9fa"];
    b_b   [label="?b",        fillcolor="#f8f9fa"];
    b_a -> b_FSa; b_FSa -> b_U; b_U -> b_B;
    b_b -> b_FSb; b_FSb -> b_B;
    b_B -> b_FE;
  }
  a_B -> b_FE [style=invis];
'

# -- 5. Base case 4: B -> B (LHS variant) ------------------------------------
render v2_base4_bb '
  subgraph cluster_lhs {
    label="LHS (LHS variant; mirror for RHS exists)"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    a_Bo [label="B_outer (false)", fillcolor="#cfe2ff"];
    a_Bi [label="B_inner (false)", fillcolor="#cfe2ff"];
    a_a  [label="?a", fillcolor="#f8f9fa"];
    a_b  [label="?b", fillcolor="#f8f9fa"];
    a_c  [label="?c", fillcolor="#f8f9fa"];
    a_a -> a_Bi; a_b -> a_Bi; a_Bi -> a_Bo; a_c -> a_Bo;
  }
  subgraph cluster_rhs {
    label="RHS"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    b_FE  [label="FE",             fillcolor="#f8d7da"];
    b_Bo  [label="B_outer (true)", fillcolor="#e0cffc"];
    b_Bi  [label="B_inner (true)", fillcolor="#e0cffc"];
    b_FSa [label="FS", fillcolor="#d1e7dd"];
    b_FSb [label="FS", fillcolor="#d1e7dd"];
    b_FSc [label="FS", fillcolor="#d1e7dd"];
    b_a   [label="?a", fillcolor="#f8f9fa"];
    b_b   [label="?b", fillcolor="#f8f9fa"];
    b_c   [label="?c", fillcolor="#f8f9fa"];
    b_a -> b_FSa; b_b -> b_FSb; b_c -> b_FSc;
    b_FSa -> b_Bi; b_FSb -> b_Bi; b_FSc -> b_Bo;
    b_Bi -> b_Bo; b_Bo -> b_FE;
  }
  a_Bo -> b_FE [style=invis];
'

# -- 6. Grow rule: FE -> outer unary ----------------------------------------
render v2_grow '
  subgraph cluster_lhs {
    label="LHS: existing FE consumed by outer unary"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    a_outer [label="?outer (Sin, false)", fillcolor="#cfe2ff"];
    a_FE    [label="?fe (FE)",            fillcolor="#f8d7da"];
    a_inner [label="?inner",              fillcolor="#fff3cd"];
    a_inner -> a_FE -> a_outer;
  }
  subgraph cluster_rhs {
    label="RHS: new FE wrapping inside-true Sin; reuses ?inner"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    b_newFE  [label="new FE",          fillcolor="#f8d7da"];
    b_inside [label="Sin (true)",      fillcolor="#e0cffc"];
    b_inner  [label="?inner",          fillcolor="#fff3cd"];
    b_inner -> b_inside -> b_newFE;
  }
  a_outer -> b_newFE [style=invis];
  labelloc=b;
  fontname="Helvetica"; fontsize=11;
  label="No new FS — ?inner is reused via e-class ref. The old FE remains in the e-graph as a smaller alternative.";
'

# -- 7. Merge rule: two FEs at a binary -------------------------------------
render v2_merge '
  subgraph cluster_lhs {
    label="LHS: outer binary takes two FE inputs"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    a_outer [label="?outer (Add, false)", fillcolor="#cfe2ff"];
    a_FEa   [label="FE_a", fillcolor="#f8d7da"];
    a_FEb   [label="FE_b", fillcolor="#f8d7da"];
    a_ia    [label="?inner_a", fillcolor="#fff3cd"];
    a_ib    [label="?inner_b", fillcolor="#fff3cd"];
    a_ia -> a_FEa; a_ib -> a_FEb;
    a_FEa -> a_outer; a_FEb -> a_outer;
  }
  subgraph cluster_rhs {
    label="RHS: one merged FE wrapping inside-true Add; both inners reused"; style="dashed,filled"; fillcolor="#fffaeb"; fontsize=11;
    b_newFE  [label="new FE", fillcolor="#f8d7da"];
    b_inside [label="Add (true)", fillcolor="#e0cffc"];
    b_ia     [label="?inner_a", fillcolor="#fff3cd"];
    b_ib     [label="?inner_b", fillcolor="#fff3cd"];
    b_ia -> b_inside; b_ib -> b_inside; b_inside -> b_newFE;
  }
  a_outer -> b_newFE [style=invis];
  labelloc=b;
  fontname="Helvetica"; fontsize=11;
  label="Both inners reused — no new FS minted. Two regions collapse into one without duplicating shared subgraphs.";
'

# -- 8. Diamond DAG result --------------------------------------------------
render v2_diamond '
  // External tensors
  a [label="a", fillcolor="#f8f9fa"];
  b [label="b", fillcolor="#f8f9fa"];

  // Single FS per distinct external tensor (the headline invariant).
  FSa [label="FS(a)", fillcolor="#d1e7dd", penwidth=2.5, color="#198754"];
  FSb [label="FS(b)", fillcolor="#d1e7dd", penwidth=2.5, color="#198754"];

  a -> FSa; b -> FSb;

  // Inside-true body, fully shared via e-class refs.
  iAdd_t [label="Add (true)\n(t = a+b)", fillcolor="#e0cffc"];
  iExp2  [label="Exp2 (true)\n(u = exp2 t)", fillcolor="#e0cffc"];
  iSin   [label="Sin (true)\n(v = sin t)",   fillcolor="#e0cffc"];
  iMul   [label="Mul (true)\n(w = u * a)",   fillcolor="#e0cffc"];
  iAdd_o [label="Add (true)\n(out = w+v)",   fillcolor="#e0cffc"];

  FSa -> iAdd_t; FSb -> iAdd_t;
  iAdd_t -> iExp2; iAdd_t -> iSin;
  iExp2 -> iMul; FSa -> iMul;          // <- a is REUSED here, same FS_a node
  iMul -> iAdd_o; iSin -> iAdd_o;

  // The full FE
  FE_full [label="FE_full", fillcolor="#f8d7da", penwidth=2.5, color="#198754"];
  iAdd_o -> FE_full;

  // out
  out [label="out", fillcolor="#f8f9fa"];
  FE_full -> out;

  labelloc=b;
  fontname="Helvetica"; fontsize=11;
  label="Diamond DAG (5 ops, 2 distinct externals {a, b}). After running pair-fuse + grow + merge:\nExactly 2 FS enodes (green-bordered) regardless of how many rules wrap a or b — congruence folds them.\nThe inner Add (t = a+b) is a single e-class shared between Exp2 and Sin paths — no duplication.\nFour FE alternatives exist in the e-graph (only the full one is shown); cost-extraction picks this one.";
'

echo "rendered $(ls v2_*.png 2>/dev/null | wc -l) v2 PNGs"
