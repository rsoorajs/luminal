use luminal::prelude::*;

fn main() {
    // Create compute graph
    let mut cx = Graph::new();
    let a = cx.tensor((3, 1));
    let b = cx.tensor((1, 4));

    let c = a.matmul(b).output();

    // Compile
    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    // Set input tensors
    rt.set_data(a, vec![1.0, 2.0, 3.0].into());
    rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0].into());

    // Run
    rt.execute(&cx.dyn_map);

    // Get output tensor
    println!("Result: {:?}", rt.get_f32(c));
}
