//! Discriminator-field experiment for binary fusion pair-rules.
//!
//! Question: does adding a Bool "inside" discriminator field to op
//! signatures actually prevent the pair-fuse rule from re-matching its
//! own RHS, or are there subtleties we're missing?
//!
//! These tests run egglog directly on a small placeholder schema
//! (no luminal infrastructure) so cascade behaviour can be observed
//! in isolation. Test plan:
//!
//!   1. Baseline (no discriminator): cascade DOES happen — confirm by
//!      counting FS enodes after N iterations and checking it grows.
//!   2. With discriminator: cascade is BLOCKED — FS count stays at 1
//!      regardless of iteration count.
//!   3. With discriminator, the rule still fires — verify the outer
//!      e-class contains the FE form.
//!   4. With discriminator on a 3-op chain — two pair rules fire once
//!      each, FS count stays at 2 (no cascade in either firing).

use luminal::prelude::egglog::{EGraph, SerializeConfig};

/// Run an egglog program; panic with a helpful message on parse/runtime
/// failure. `(check ...)` failures inside the program surface as runtime
/// errors here.
fn run(program: &str) -> EGraph {
    let mut egraph = EGraph::default();
    let cmds = egraph
        .parser
        .get_program_from_string(None, program)
        .expect("egglog parse error");
    egraph.run_program(cmds).expect("egglog runtime error");
    egraph
}

/// Count enodes in the e-graph whose op label matches `op`.
fn count_op(egraph: &EGraph, op: &str) -> usize {
    let s = egraph.serialize(SerializeConfig {
        root_eclasses: vec![],
        max_functions: None,
        include_temporary_functions: false,
        max_calls_per_function: None,
    });
    s.egraph.nodes.iter().filter(|(_, n)| n.op == op).count()
}

#[test]
fn baseline_no_discriminator_cascades() {
    // Pair-fuse rule WITHOUT discriminator. Pattern U2(U1(?x)) re-matches
    // its own freshly-minted RHS (U2_sep(U1_sep(FS(?x)))) because that
    // structure is itself a U2-of-U1, this time with ?x bound to FS(in).
    // Each iteration mints another FS layer.
    let program = r#"
        (datatype Term
            (Var String)
            (Sqrt Term)
            (Sin Term)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?u1 (Sqrt ?in))
             (= ?u2 (Sin ?u1)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sqrt ?fs))
             (let ?u2s (Sin ?u1s))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main
            :name "pair-fuse")

        (let x (Var "x"))
        (let outer (Sin (Sqrt x)))

        (run-schedule (repeat 5 (run main)))
    "#;
    let egraph = run(program);
    let fs_count = count_op(&egraph, "FS");
    eprintln!("[baseline] FS enodes after 5 iters: {fs_count}");
    assert!(
        fs_count > 1,
        "expected cascade to mint multiple FS enodes (>1), got {fs_count}"
    );
}

#[test]
fn discriminator_blocks_cascade() {
    // With a Bool "inside" field on each op kind, the pair-fuse rule's
    // LHS pattern requires ?inside = false. The RHS produces ops with
    // ?inside = true. The pattern can never match its own output.
    // Expectation: exactly 1 FS enode regardless of iteration count.
    let program = r#"
        (datatype Term
            (Var String)
            (Sqrt Term bool)
            (Sin Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?u1 (Sqrt ?in false))
             (= ?u2 (Sin ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sqrt ?fs true))
             (let ?u2s (Sin ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main
            :name "pair-fuse")

        (let x (Var "x"))
        (let outer (Sin (Sqrt x false) false))

        (run-schedule (repeat 5 (run main)))
    "#;
    let egraph = run(program);
    let fs_count = count_op(&egraph, "FS");
    eprintln!("[discriminator] FS enodes after 5 iters: {fs_count}");
    assert_eq!(
        fs_count, 1,
        "expected exactly 1 FS enode (no cascade), got {fs_count}"
    );
}

#[test]
fn discriminator_preserves_legitimate_fusion() {
    // Sanity: the discriminator must not block the rule's initial firing.
    // After running, outer's e-class should contain the full fused form.
    // egglog's own (check (= ...)) succeeds iff outer and the fused term
    // are in the same e-class. If the rule never fired, this fails.
    let program = r#"
        (datatype Term
            (Var String)
            (Sqrt Term bool)
            (Sin Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?u1 (Sqrt ?in false))
             (= ?u2 (Sin ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sqrt ?fs true))
             (let ?u2s (Sin ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main
            :name "pair-fuse")

        (let x (Var "x"))
        (let outer (Sin (Sqrt x false) false))

        (run-schedule (repeat 5 (run main)))

        (check (= outer (FE (Sin (Sqrt (FS x) true) true))))
    "#;
    let _ = run(program);
}

#[test]
fn discriminator_three_op_chain_stable() {
    // 3-op chain Sin(Sqrt(Recip(x))) with two pair-fuse rules:
    //   - (Recip, Sqrt) fires once on the inner pair
    //   - (Sqrt, Sin) fires once on the outer pair
    // Each firing mints its own FS. With the discriminator, neither
    // rule re-fires on its own output. Expectation: exactly 2 FS
    // enodes after `repeat 10`.
    let program = r#"
        (datatype Term
            (Var String)
            (Recip Term bool)
            (Sqrt Term bool)
            (Sin Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?u1 (Recip ?in false))
             (= ?u2 (Sqrt ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Recip ?fs true))
             (let ?u2s (Sqrt ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main
            :name "pair-fuse-Recip-Sqrt")

        (rule
            ((= ?u1 (Sqrt ?in false))
             (= ?u2 (Sin ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sqrt ?fs true))
             (let ?u2s (Sin ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main
            :name "pair-fuse-Sqrt-Sin")

        (let x (Var "x"))
        (let outer (Sin (Sqrt (Recip x false) false) false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs_count = count_op(&egraph, "FS");
    let fe_count = count_op(&egraph, "FE");
    eprintln!("[3-op chain] FS={fs_count} FE={fe_count}");
    assert_eq!(
        fs_count, 2,
        "expected 2 FS enodes (one per pair firing), got {fs_count}"
    );
    assert_eq!(
        fe_count, 2,
        "expected 2 FE enodes (one per pair firing), got {fe_count}"
    );
}

// ----------------------------------------------------------------------------
// Base case 2: binary -> unary pair fusion (e.g. Exp(Add(a, b))).
// The pair-fuse rule mints two FS markers (one per binary input), a fused
// inside-copy of both ops, and an FE on top, then unions the outer unary
// with the FE. Same discriminator idiom as base case 1.
// ----------------------------------------------------------------------------

#[test]
fn discriminator_binary_unary_blocks_cascade() {
    // Exp(Add(a, b)) with distinct inputs. After firing, expected:
    //   - 2 FS enodes (one per binary input)
    //   - 1 FE enode (one per region)
    //   - exactly one fused-inside Add and one fused-inside Exp
    // Stable across `repeat 10` because the pattern requires
    // inside=false on both ops, and the RHS produces inside=true.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?exp (Exp ?add false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?exp_f (Exp ?add_f true))
             (let ?fe    (FE ?exp_f))
             (union ?exp ?fe))
            :ruleset main
            :name "pair-fuse-Add-Exp")

        (let a (Var "a"))
        (let b (Var "b"))
        (let outer (Exp (Add a b false) false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[binary-unary] FS={fs} FE={fe}");
    assert_eq!(fs, 2, "expected 2 FS (one per binary input), got {fs}");
    assert_eq!(fe, 1, "expected 1 FE, got {fe}");
}

#[test]
fn discriminator_binary_unary_preserves_fusion() {
    // Sanity: the merged outer e-class contains the full fused form.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?exp (Exp ?add false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?exp_f (Exp ?add_f true))
             (let ?fe    (FE ?exp_f))
             (union ?exp ?fe))
            :ruleset main
            :name "pair-fuse-Add-Exp")

        (let a (Var "a"))
        (let b (Var "b"))
        (let outer (Exp (Add a b false) false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer (FE (Exp (Add (FS a) (FS b) true) true))))
    "#;
    let _ = run(program);
}

// ----------------------------------------------------------------------------
// Base case 3: unary -> binary pair fusion (e.g. Add(Exp(a), b)).
// Asymmetric vs base case 2: only the binary's *external* input needs an FS
// at the binary level. The unary's input gets its own FS one level deeper;
// the unary's output flows into the fused-inside binary directly. Two FSes
// total (one per distinct external tensor — a and b).
// ----------------------------------------------------------------------------

#[test]
fn discriminator_unary_binary_lhs_blocks_cascade() {
    // Add(Exp(a), b) with distinct inputs. Expected:
    //   - 2 FS (one per external tensor: a and b)
    //   - 1 FE
    // Stable across `repeat 10`.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?exp ?b false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?exp_f ?fs_b true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-LHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let outer (Add (Exp a false) b false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[unary-binary lhs] FS={fs} FE={fe}");
    assert_eq!(fs, 2, "expected 2 FS, got {fs}");
    assert_eq!(fe, 1, "expected 1 FE, got {fe}");
}

#[test]
fn discriminator_unary_binary_lhs_preserves_fusion() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?exp ?b false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?exp_f ?fs_b true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-LHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let outer (Add (Exp a false) b false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer (FE (Add (Exp (FS a) true) (FS b) true))))
    "#;
    let _ = run(program);
}

#[test]
fn discriminator_unary_binary_lhs_same_input_collapses_fs() {
    // Add(Exp(x), x) — the binary's external RHS reuses the unary's source.
    // Both FS enodes wrap x's e-class, so hash consing should fold them to
    // a single FS.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?exp ?b false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?exp_f ?fs_b true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-LHS")

        (let x (Var "x"))
        (let outer (Add (Exp x false) x false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    eprintln!("[unary-binary lhs same input] FS={fs}");
    assert_eq!(fs, 1, "expected 1 FS (shared source), got {fs}");
}

// ----------------------------------------------------------------------------
// Base case 3 mirror: unary on the RHS of the binary, e.g. Add(b, Exp(a)).
// Necessary because Add ?lhs ?rhs is positional in egglog patterns even when
// the underlying op is mathematically commutative — without a separate rule,
// the LHS-only pattern would silently miss this configuration. Same FS count
// (one per distinct external tensor) and same cascade-blocked behaviour.
// ----------------------------------------------------------------------------

#[test]
fn discriminator_unary_binary_rhs_blocks_cascade() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?b ?exp false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?fs_b ?exp_f true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-RHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let outer (Add b (Exp a false) false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[unary-binary rhs] FS={fs} FE={fe}");
    assert_eq!(fs, 2, "expected 2 FS, got {fs}");
    assert_eq!(fe, 1, "expected 1 FE, got {fe}");
}

#[test]
fn discriminator_unary_binary_rhs_preserves_fusion() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?b ?exp false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?fs_b ?exp_f true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-RHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let outer (Add b (Exp a false) false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer (FE (Add (FS b) (Exp (FS a) true) true))))
    "#;
    let _ = run(program);
}

#[test]
fn discriminator_unary_binary_rhs_same_input_collapses_fs() {
    // Add(x, Exp(x)) — binary's external LHS reuses the unary's source.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?b ?exp false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?fs_b ?exp_f true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-RHS")

        (let x (Var "x"))
        (let outer (Add x (Exp x false) false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    eprintln!("[unary-binary rhs same input] FS={fs}");
    assert_eq!(fs, 1, "expected 1 FS (shared source), got {fs}");
}

// ----------------------------------------------------------------------------
// Coexistence: both LHS and RHS rules in the same program. Confirms the two
// rules don't interfere — Add(Exp(a), b) only fires the LHS rule, Add(b, Exp(a))
// only fires the RHS rule, no double-firing on a single configuration.
// ----------------------------------------------------------------------------

#[test]
fn discriminator_unary_binary_both_directions_coexist() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?exp ?b false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?exp_f ?fs_b true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-LHS")

        (rule
            ((= ?exp (Exp ?a false))
             (= ?add (Add ?b ?exp false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?exp_f (Exp ?fs_a true))
             (let ?add_f (Add ?fs_b ?exp_f true))
             (let ?fe    (FE ?add_f))
             (union ?add ?fe))
            :ruleset main
            :name "pair-fuse-Exp-Add-RHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let lhs_term (Add (Exp a false) b false))
        (let rhs_term (Add b (Exp a false) false))

        (run-schedule (repeat 10 (run main)))

        ; Both terms should reach their respective fused forms
        (check (= lhs_term (FE (Add (Exp (FS a) true) (FS b) true))))
        (check (= rhs_term (FE (Add (FS b) (Exp (FS a) true) true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[both dirs coexist] FS={fs} FE={fe}");
    // Two terms × 2 distinct external tensors {a, b} = 2 FS via hash consing.
    // Two terms × 1 FE each = 2 FE.
    assert_eq!(fs, 2, "expected 2 shared FS enodes for {{a, b}}, got {fs}");
    assert_eq!(fe, 2, "expected 2 FE enodes (one per term), got {fe}");
}

#[test]
fn discriminator_binary_unary_same_input_collapses_fs() {
    // Exp(Add(x, x)) — both binary inputs are the same e-class.
    // Hash consing should fold the two FS enodes (FS(x) and FS(x))
    // into a single shared enode. Design invariant: "one FS per
    // distinct external tensor", not per edge.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Exp Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?exp (Exp ?add false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?exp_f (Exp ?add_f true))
             (let ?fe    (FE ?exp_f))
             (union ?exp ?fe))
            :ruleset main
            :name "pair-fuse-Add-Exp")

        (let x (Var "x"))
        (let outer (Exp (Add x x false) false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    eprintln!("[binary-unary same input] FS={fs}");
    assert_eq!(
        fs, 1,
        "expected 1 FS (both binary inputs share an e-class), got {fs}"
    );
}

// ----------------------------------------------------------------------------
// Base case 4: binary -> binary pair fusion (e.g. Mul(Add(a, b), c)).
// Three external inputs cross the region boundary, so 3 FSes total. The
// fused inner is an Add feeding a Mul, both inside=true. FE on top, unioned
// with the outer Mul. RHS mirror exists for Mul(c, Add(a, b)).
// ----------------------------------------------------------------------------

#[test]
fn discriminator_binary_binary_lhs_blocks_cascade() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?add ?c false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?add_f ?fs_c true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-LHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let c (Var "c"))
        (let outer (Mul (Add a b false) c false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[binary-binary lhs] FS={fs} FE={fe}");
    assert_eq!(fs, 3, "expected 3 FS (one per external tensor), got {fs}");
    assert_eq!(fe, 1, "expected 1 FE, got {fe}");
}

#[test]
fn discriminator_binary_binary_lhs_preserves_fusion() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?add ?c false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?add_f ?fs_c true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-LHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let c (Var "c"))
        (let outer (Mul (Add a b false) c false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer
                  (FE (Mul (Add (FS a) (FS b) true) (FS c) true))))
    "#;
    let _ = run(program);
}

#[test]
fn discriminator_binary_binary_rhs_blocks_cascade() {
    // Mirror: Mul(c, Add(a, b)) — inner binary on the RHS of the outer.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?c ?add false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?fs_c ?add_f true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-RHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let c (Var "c"))
        (let outer (Mul c (Add a b false) false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[binary-binary rhs] FS={fs} FE={fe}");
    assert_eq!(fs, 3, "expected 3 FS, got {fs}");
    assert_eq!(fe, 1, "expected 1 FE, got {fe}");
}

#[test]
fn discriminator_binary_binary_rhs_preserves_fusion() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?c ?add false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?fs_c ?add_f true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-RHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let c (Var "c"))
        (let outer (Mul c (Add a b false) false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer
                  (FE (Mul (FS c) (Add (FS a) (FS b) true) true))))
    "#;
    let _ = run(program);
}

#[test]
fn discriminator_binary_binary_all_inputs_share_collapses_fs() {
    // Mul(Add(x, x), x) — all three external inputs are the same e-class.
    // Hash consing should fold all three FS enodes into one.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?add ?c false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?add_f ?fs_c true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-LHS")

        (let x (Var "x"))
        (let outer (Mul (Add x x false) x false))

        (run-schedule (repeat 10 (run main)))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    eprintln!("[binary-binary all share] FS={fs}");
    assert_eq!(
        fs, 1,
        "expected 1 FS (all three external positions share an e-class), got {fs}"
    );
}

#[test]
fn discriminator_binary_binary_both_directions_coexist() {
    // Both LHS and RHS rules in the same program with both term shapes.
    // Hash consing should fold the FS-wraps of {a, b, c} across both rules.
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?add ?c false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?add_f ?fs_c true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-LHS")

        (rule
            ((= ?add (Add ?a ?b false))
             (= ?mul (Mul ?c ?add false)))
            ((let ?fs_a  (FS ?a))
             (let ?fs_b  (FS ?b))
             (let ?fs_c  (FS ?c))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?mul_f (Mul ?fs_c ?add_f true))
             (let ?fe    (FE ?mul_f))
             (union ?mul ?fe))
            :ruleset main
            :name "pair-fuse-Add-Mul-RHS")

        (let a (Var "a"))
        (let b (Var "b"))
        (let c (Var "c"))
        (let lhs_term (Mul (Add a b false) c false))
        (let rhs_term (Mul c (Add a b false) false))

        (run-schedule (repeat 10 (run main)))

        (check (= lhs_term (FE (Mul (Add (FS a) (FS b) true) (FS c) true))))
        (check (= rhs_term (FE (Mul (FS c) (Add (FS a) (FS b) true) true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[binary-binary both dirs] FS={fs} FE={fe}");
    assert_eq!(fs, 3, "expected 3 shared FS for {{a, b, c}}, got {fs}");
    assert_eq!(fe, 2, "expected 2 FE (one per term), got {fe}");
}

// ============================================================================
// GROW RULES (Trial 1)
// ----------------------------------------------------------------------------
// A grow rule extends an existing FE region past an adjacent compatible op.
// Discriminator on the OUTER op's pattern (`inside=false`) blocks cascade
// the same way as in pair-fuse: the rule's RHS produces an inside=true
// fused copy which can never re-match the LHS.
//
// Open question: how many FE alternatives accumulate per chain?
// ============================================================================

#[test]
fn grow_unary_three_op_chain_blocks_cascade_and_finds_full_fusion() {
    // Chain: Recip(Sin(Sqrt(x))). One pair-fuse rule (Sqrt-Sin), plus
    // a grow rule that absorbs an outer Recip into an existing FE.
    // Expected after running:
    //   - Pair-fuse fires once on (Sqrt, Sin) -> FE_12 ({Sqrt, Sin})
    //   - Grow fires once on FE_12 + Recip -> FE_123 ({Sqrt, Sin, Recip})
    //   - 1 FS (for x) total
    //   - 2 FE (one per region size)
    //   - The full 3-op fused form is reachable in `outer`'s e-class
    let program = r#"
        (datatype Term
            (Var String)
            (Sqrt Term bool)
            (Sin Term bool)
            (Recip Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        ; Pair-fuse: Sqrt-Sin
        (rule
            ((= ?u1 (Sqrt ?in false))
             (= ?u2 (Sin ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sqrt ?fs true))
             (let ?u2s (Sin ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main
            :name "pair-fuse-Sqrt-Sin")

        ; Grow: FE -> Recip (forward chain extension)
        (rule
            ((= ?fe (FE ?inner))
             (= ?outer (Recip ?fe false)))
            ((let ?inside (Recip ?inner true))
             (let ?new_fe (FE ?inside))
             (union ?outer ?new_fe))
            :ruleset main
            :name "grow-FE-Recip")

        (let x (Var "x"))
        (let outer (Recip (Sin (Sqrt x false) false) false))

        (run-schedule (repeat 10 (run main)))

        ; The full 3-op fused form should be reachable
        (check (= outer (FE (Recip (Sin (Sqrt (FS x) true) true) true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[grow unary 3-op] FS={fs} FE={fe}");
    assert_eq!(fs, 1, "expected 1 FS for the single external x, got {fs}");
    assert_eq!(
        fe, 2,
        "expected 2 FE (small region {{Sqrt,Sin}} + grown region {{Sqrt,Sin,Recip}}), got {fe}"
    );
}

#[test]
fn grow_unary_four_op_chain_quadratic_alternatives() {
    // Chain: Log(Recip(Sin(Sqrt(x)))). With every pair-fuse rule and
    // every grow rule available, the e-graph accumulates FEs for
    // every possible contiguous subchain of length >= 2.
    //   N=4 chain -> subchains: (1,2) (2,3) (3,4) (1,2,3) (2,3,4) (1,2,3,4) = 6
    // This is the quadratic-in-chain-length blowup. Each FE is a
    // distinct semantic alternative (different region output value),
    // so they're not "duplicates" in the cascade sense — they're
    // legitimate fusion choices that cost-based extraction must rank.
    let program = r#"
        (datatype Term
            (Var String)
            (Sqrt Term bool)
            (Sin Term bool)
            (Recip Term bool)
            (Log Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        ; Pair-fuse rules for every adjacent pair in the chain
        (rule
            ((= ?u1 (Sqrt ?in false))
             (= ?u2 (Sin ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sqrt ?fs true))
             (let ?u2s (Sin ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main :name "pf-Sqrt-Sin")
        (rule
            ((= ?u1 (Sin ?in false))
             (= ?u2 (Recip ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sin ?fs true))
             (let ?u2s (Recip ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main :name "pf-Sin-Recip")
        (rule
            ((= ?u1 (Recip ?in false))
             (= ?u2 (Log ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Recip ?fs true))
             (let ?u2s (Log ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main :name "pf-Recip-Log")

        ; Grow rules: one per outer op kind
        (rule
            ((= ?fe (FE ?inner))
             (= ?outer (Sin ?fe false)))
            ((let ?inside (Sin ?inner true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "grow-Sin")
        (rule
            ((= ?fe (FE ?inner))
             (= ?outer (Recip ?fe false)))
            ((let ?inside (Recip ?inner true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "grow-Recip")
        (rule
            ((= ?fe (FE ?inner))
             (= ?outer (Log ?fe false)))
            ((let ?inside (Log ?inner true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "grow-Log")

        (let x (Var "x"))
        (let outer (Log (Recip (Sin (Sqrt x false) false) false) false))

        (run-schedule (repeat 10 (run main)))

        ; Full 4-op fused form must be reachable
        (check (= outer
                  (FE (Log (Recip (Sin (Sqrt (FS x) true) true) true) true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[grow unary 4-op] FS={fs} FE={fe}");
    // Stability under repeat 10 is the key: no cascade, the count is
    // determined purely by the chain shape.
    assert!(fe >= 6, "expected at least 6 FE alternatives, got {fe}");
    // FS count: 1 for x (used by all left-anchored regions) plus one
    // per intermediate value that some pair-fuse rule treats as an
    // external boundary. Depends on which pair-fuse fires.
    assert!(fs >= 1 && fs <= 4, "FS count out of expected range: {fs}");
}

#[test]
fn grow_binary_lhs_extends_region_past_add() {
    // Region {Exp} fused via pair-fuse-Exp-Add could have started as
    // a unary-only fusion; here we test the symmetric case: pair-fuse
    // creates a region wrapping a unary, and grow extends it past an
    // outer binary that consumes the unary's output as its LHS.
    //
    // Chain: Add(Sqrt(Sin(x)), b). Pair-fuse (Sin, Sqrt) creates
    // FE_inner. Grow Add-LHS extends past the Add.
    let program = r#"
        (datatype Term
            (Var String)
            (Sin Term bool)
            (Sqrt Term bool)
            (Add Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?u1 (Sin ?in false))
             (= ?u2 (Sqrt ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sin ?fs true))
             (let ?u2s (Sqrt ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main :name "pair-fuse-Sin-Sqrt")

        ; Grow FE -> Add, FE on LHS, RHS external
        (rule
            ((= ?fe (FE ?inner))
             (= ?outer (Add ?fe ?b false)))
            ((let ?fs_b (FS ?b))
             (let ?inside (Add ?inner ?fs_b true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "grow-FE-Add-LHS")

        (let x (Var "x"))
        (let b (Var "b"))
        (let outer (Add (Sqrt (Sin x false) false) b false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer
                  (FE (Add (Sqrt (Sin (FS x) true) true) (FS b) true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[grow binary lhs] FS={fs} FE={fe}");
    assert_eq!(fs, 2, "expected 2 FS (x and b), got {fs}");
    assert_eq!(fe, 2, "expected 2 FE (small + grown), got {fe}");
}

#[test]
fn merge_two_fes_at_binary() {
    // Diamond top: Add(Sqrt(Sin(a)), Recip(b)) — both branches are
    // their own pair-fused regions, and the outer Add merges them.
    //
    // Pair-fuse fires twice (once per branch), creating two FEs.
    // Merge rule fires once on the outer Add, producing one big FE.
    let program = r#"
        (datatype Term
            (Var String)
            (Sin Term bool)
            (Sqrt Term bool)
            (Recip Term bool)
            (Add Term Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        (rule
            ((= ?u1 (Sin ?in false))
             (= ?u2 (Sqrt ?u1 false)))
            ((let ?fs (FS ?in))
             (let ?u1s (Sin ?fs true))
             (let ?u2s (Sqrt ?u1s true))
             (let ?fe (FE ?u2s))
             (union ?u2 ?fe))
            :ruleset main :name "pf-Sin-Sqrt")

        ; Pair-fuse Recip alone needs another pair partner; here we
        ; manually seed the right branch with a synthetic rule so the
        ; merge has two FEs to consume. In practice, the right branch
        ; would have been pair-fused with something downstream too.
        (rule
            ((= ?u (Recip ?in false))
             (Var ?n))
            ((let ?fs (FS ?in))
             (let ?us (Recip ?fs true))
             (let ?fe (FE ?us))
             (union ?u ?fe))
            :ruleset main :name "seed-Recip")

        ; Merge: B(FE_a, FE_b) -> FE(B(inner_a, inner_b))
        (rule
            ((= ?fe_a (FE ?inner_a))
             (= ?fe_b (FE ?inner_b))
             (= ?outer (Add ?fe_a ?fe_b false)))
            ((let ?inside (Add ?inner_a ?inner_b true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "merge-FEs-Add")

        (let a (Var "a"))
        (let b (Var "b"))
        (let lhs_branch (Sqrt (Sin a false) false))
        (let rhs_branch (Recip b false))
        (let outer (Add lhs_branch rhs_branch false))

        (run-schedule (repeat 10 (run main)))

        (check (= outer
                  (FE (Add (Sqrt (Sin (FS a) true) true)
                           (Recip (FS b) true)
                           true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[merge at binary] FS={fs} FE={fe}");
    assert_eq!(fs, 2, "expected 2 FS (a and b), got {fs}");
    // Three FEs expected: FE_left ({Sin, Sqrt}), FE_right ({Recip} alone
    // via the synthetic seed), FE_merged (full Add region)
    assert_eq!(fe, 3, "expected 3 FE (two branches + merged), got {fe}");
}

// ============================================================================
// DIAMOND DAG — the original failing case from
// `crates/luminal_cuda_lite/src/tests/fusion.rs::test_diamond_dag_fuses` and
// the `#[ignore]`'d `test_fused_region_starts_match_distinct_external_tensors`.
// ----------------------------------------------------------------------------
// Source program:
//   t   = a + b
//   u   = exp2(t)
//   v   = sin(t)
//   w   = u * a       <- reuses external `a`
//   out = w + v
//
// Five ops, two distinct external tensors {a, b}. Reuse of `a` was the
// source of the FS_a1 / FS_a2 spurious-duplicate problem in the old rule
// set: each rule firing minted a fresh FS for `a` and they never unified.
//
// With Trial 1 (discriminator + grow + merge), congruence/hash-consing
// folds every FS-wrap of `a` to a single enode regardless of which rule
// minted it. Expected outcome:
//   - Full 5-op fused region is reachable in `outer`'s e-class
//   - Exactly 2 FS enodes in the e-graph (for {a, b})
//   - Smaller FE alternatives also coexist (each is a valid sub-fusion)
// ============================================================================

#[test]
fn diamond_dag_full_fusion_reachable_with_two_fs() {
    let program = r#"
        (datatype Term
            (Var String)
            (Add Term Term bool)
            (Mul Term Term bool)
            (Exp2 Term bool)
            (Sin Term bool)
            (FS Term)
            (FE Term))

        (ruleset main)

        ; Pair-fuse: Add -> Exp2
        (rule
            ((= ?add (Add ?a ?b false))
             (= ?exp2 (Exp2 ?add false)))
            ((let ?fs_a (FS ?a))
             (let ?fs_b (FS ?b))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?exp2_f (Exp2 ?add_f true))
             (union ?exp2 (FE ?exp2_f)))
            :ruleset main :name "pf-Add-Exp2")

        ; Pair-fuse: Add -> Sin
        (rule
            ((= ?add (Add ?a ?b false))
             (= ?sin (Sin ?add false)))
            ((let ?fs_a (FS ?a))
             (let ?fs_b (FS ?b))
             (let ?add_f (Add ?fs_a ?fs_b true))
             (let ?sin_f (Sin ?add_f true))
             (union ?sin (FE ?sin_f)))
            :ruleset main :name "pf-Add-Sin")

        ; Grow FE -> Mul, FE on LHS, RHS external
        (rule
            ((= ?fe (FE ?inner))
             (= ?outer (Mul ?fe ?b false)))
            ((let ?fs_b (FS ?b))
             (let ?inside (Mul ?inner ?fs_b true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "grow-FE-Mul-LHS")

        ; Merge: outer Add consuming two FEs collapses both regions
        (rule
            ((= ?fe_a (FE ?inner_a))
             (= ?fe_b (FE ?inner_b))
             (= ?outer (Add ?fe_a ?fe_b false)))
            ((let ?inside (Add ?inner_a ?inner_b true))
             (union ?outer (FE ?inside)))
            :ruleset main :name "merge-Add")

        ; Build the diamond DAG
        (let a (Var "a"))
        (let b (Var "b"))
        (let t (Add a b false))
        (let u (Exp2 t false))
        (let v (Sin t false))
        (let w (Mul u a false))
        (let outer (Add w v false))

        (run-schedule (repeat 10 (run main)))

        ; The full 5-op fused form must be in outer's e-class.
        ; Note: FS a appears 3 times in this term (two Add_inside LHS slots
        ; plus the Mul_inside RHS slot), but congruence makes them all one
        ; e-class. Same for FS b (used by both Add_inside copies).
        (check (= outer
                  (FE (Add
                        (Mul
                          (Exp2 (Add (FS a) (FS b) true) true)
                          (FS a)
                          true)
                        (Sin (Add (FS a) (FS b) true) true)
                        true))))
    "#;
    let egraph = run(program);
    let fs = count_op(&egraph, "FS");
    let fe = count_op(&egraph, "FE");
    eprintln!("[diamond] FS={fs} FE={fe}");

    // The headline invariant: exactly 2 FS for the diamond's 2 distinct
    // external tensors {a, b}. This is what the old rule set blew up to
    // 3 (with FS_a1, FS_a2 unmerged), and it's what the `#[ignore]`'d
    // test was waiting on.
    assert_eq!(
        fs, 2,
        "diamond should have exactly 2 FS for {{a, b}}, got {fs}"
    );

    // FE alternatives:
    //   FE_AE   — pair-fuse Add-Exp2  ({Add, Exp2})
    //   FE_AS   — pair-fuse Add-Sin   ({Add, Sin})
    //   FE_AEM  — grow over Mul       ({Add, Exp2, Mul})
    //   FE_full — merge at outer Add  (full diamond)
    assert_eq!(fe, 4, "expected 4 FE alternatives, got {fe}");
}

#[test]
fn diamond_dag_stable_under_repeat() {
    // Sanity: repeat 5 vs repeat 20 should give identical FS/FE counts.
    // If the discriminator slipped through somewhere and produced
    // cascade behaviour on the diamond's reused-input shape, we'd see
    // counts grow with iteration count.
    let program_template = |iters: u32| {
        format!(
            r#"
            (datatype Term
                (Var String)
                (Add Term Term bool)
                (Mul Term Term bool)
                (Exp2 Term bool)
                (Sin Term bool)
                (FS Term)
                (FE Term))

            (ruleset main)

            (rule
                ((= ?add (Add ?a ?b false))
                 (= ?exp2 (Exp2 ?add false)))
                ((let ?fs_a (FS ?a))
                 (let ?fs_b (FS ?b))
                 (let ?add_f (Add ?fs_a ?fs_b true))
                 (let ?exp2_f (Exp2 ?add_f true))
                 (union ?exp2 (FE ?exp2_f)))
                :ruleset main :name "pf-Add-Exp2")
            (rule
                ((= ?add (Add ?a ?b false))
                 (= ?sin (Sin ?add false)))
                ((let ?fs_a (FS ?a))
                 (let ?fs_b (FS ?b))
                 (let ?add_f (Add ?fs_a ?fs_b true))
                 (let ?sin_f (Sin ?add_f true))
                 (union ?sin (FE ?sin_f)))
                :ruleset main :name "pf-Add-Sin")
            (rule
                ((= ?fe (FE ?inner))
                 (= ?outer (Mul ?fe ?b false)))
                ((let ?fs_b (FS ?b))
                 (let ?inside (Mul ?inner ?fs_b true))
                 (union ?outer (FE ?inside)))
                :ruleset main :name "grow-FE-Mul-LHS")
            (rule
                ((= ?fe_a (FE ?inner_a))
                 (= ?fe_b (FE ?inner_b))
                 (= ?outer (Add ?fe_a ?fe_b false)))
                ((let ?inside (Add ?inner_a ?inner_b true))
                 (union ?outer (FE ?inside)))
                :ruleset main :name "merge-Add")

            (let a (Var "a"))
            (let b (Var "b"))
            (let t (Add a b false))
            (let u (Exp2 t false))
            (let v (Sin t false))
            (let w (Mul u a false))
            (let outer (Add w v false))

            (run-schedule (repeat {iters} (run main)))
            "#
        )
    };

    let g_short = run(&program_template(5));
    let g_long = run(&program_template(20));

    let fs_short = count_op(&g_short, "FS");
    let fs_long = count_op(&g_long, "FS");
    let fe_short = count_op(&g_short, "FE");
    let fe_long = count_op(&g_long, "FE");

    eprintln!("[diamond stable] short(repeat 5): FS={fs_short} FE={fe_short}");
    eprintln!("[diamond stable]  long(repeat 20): FS={fs_long}  FE={fe_long}");

    assert_eq!(
        fs_short, fs_long,
        "FS count must be stable under iteration: {fs_short} vs {fs_long}"
    );
    assert_eq!(
        fe_short, fe_long,
        "FE count must be stable under iteration: {fe_short} vs {fe_long}"
    );
}
