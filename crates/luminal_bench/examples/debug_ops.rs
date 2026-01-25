//! Debug script to locate which operation causes "No valid graphs present in the e-graph!"
//!
//! Run with:
//! ```bash
//! cargo run -p luminal_bench --features metal --example debug_ops
//! ```

use luminal::prelude::*;
use luminal::op::Runtime;
use luminal_metal::runtime::MetalRuntime;

fn test_op(name: &str, build_fn: impl FnOnce(&mut Graph)) -> bool {
    println!("\n=== Testing: {} ===", name);

    let mut cx = Graph::default();
    build_fn(&mut cx);

    cx.build_search_space::<MetalRuntime>();
    let rt = MetalRuntime::initialize(());

    // Try to search - this is where it might fail
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cx.search(rt, 1)
    }));

    match result {
        Ok(_) => {
            println!("  ✅ PASS");
            true
        }
        Err(_) => {
            println!("  ❌ FAIL - No valid graph found!");
            false
        }
    }
}

fn main() {
    println!("=== Debug: Locating unsupported operations ===\n");

    // Level 1: Basic operations
    println!("\n--- Level 1: Basic Operations ---");

    test_op("Mul (x * x)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = (x * x).output();
    });

    test_op("Add (x + 1.0)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = (x + 1.0).output();
    });

    test_op("Sqrt", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.sqrt().output();
    });

    test_op("Recip", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.reciprocal().output();
    });

    // Level 2: Reduce operations
    println!("\n--- Level 2: Reduce Operations ---");

    test_op("Sum reduce (axis=1)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.sum(1).output();
    });

    test_op("Max reduce (axis=1)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.max(1).output();
    });

    // Level 3: Mean (requires sum + division)
    println!("\n--- Level 3: Mean Operation ---");

    test_op("Mean (axis=1)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.mean(1).output();
    });

    // Level 4: Broadcast/Expand
    println!("\n--- Level 4: Broadcast Operations ---");

    test_op("Sum + expand_to_shape_on_axes", |cx| {
        let x = cx.tensor((128, 128));
        let summed = x.sum(1);  // shape: (128,)
        let _ = summed.expand_to_shape_on_axes(x.shape, 1).output();  // broadcast back
    });

    test_op("Mean + expand_to_shape_on_axes", |cx| {
        let x = cx.tensor((128, 128));
        let mean = x.mean(1);
        let _ = mean.expand_to_shape_on_axes(x.shape, 1).output();
    });

    // Level 5: Subtraction (a - b = a + b * -1)
    println!("\n--- Level 5: Subtraction ---");

    test_op("Subtraction (x - y)", |cx| {
        let x = cx.tensor((128, 128));
        let y = cx.tensor((128, 128));
        let _ = (x - y).output();
    });

    test_op("x - x.mean().expand() (mean_norm core)", |cx| {
        let x = cx.tensor((128, 128));
        let mean = x.mean(1).expand_to_shape_on_axes(x.shape, 1);
        let _ = (x - mean).output();
    });

    // Level 6: mean_norm
    println!("\n--- Level 6: mean_norm ---");

    test_op("mean_norm (full)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.mean_norm(1).output();
    });

    // Level 7: std_norm components
    println!("\n--- Level 7: std_norm components ---");

    test_op("(x*x).mean().sqrt().recip()", |cx| {
        let x = cx.tensor((128, 128));
        let _ = (x * x).mean(1).sqrt().reciprocal().output();
    });

    test_op("variance + expand + mul", |cx| {
        let x = cx.tensor((128, 128));
        let var = ((x * x).mean(1) + 1e-5).sqrt().reciprocal();
        let _ = (var.expand_to_shape_on_axes(x.shape, 1) * x).output();
    });

    // Level 8: Full layer_norm
    println!("\n--- Level 8: Full layer_norm ---");

    test_op("layer_norm (full)", |cx| {
        let x = cx.tensor((128, 128));
        let _ = x.layer_norm(1, 1e-5).output();
    });

    println!("\n=== Debug Complete ===");
}
