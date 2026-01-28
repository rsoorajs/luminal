use std::env;
use std::process::Command;

fn main() {
    // Only rerun this build script if anything in the setup directory changes
    println!("cargo:rerun-if-changed=setup");
    println!("cargo:rerun-if-changed=.cargo/config.toml");

    // Run the setup script with uv
    let status = Command::new("uv")
        .args([
            "run",
            "--script",
            "setup/setup.py",
            &env::var("LUMINAL_EXAMPLE_HF_MODEL").unwrap(),
        ])
        .status()
        .expect("Failed to execute setup script. Make sure 'uv' is installed.");

    if !status.success() {
        panic!("Setup script failed with status: {}", status);
    }
}
