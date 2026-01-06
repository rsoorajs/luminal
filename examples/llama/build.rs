use std::process::Command;

fn main() {
    // Only rerun this build script if the setup script itself changes
    println!("cargo:rerun-if-changed=setup/setup.py");

    // Run the setup script with uv
    let status = Command::new("uv")
        .args(["run", "--script", "setup/setup.py"])
        .status()
        .expect("Failed to execute setup script. Make sure 'uv' is installed.");

    if !status.success() {
        panic!("Setup script failed with status: {}", status);
    }
}
