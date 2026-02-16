"""Test configuration."""
# Enable automatic Rust rebuilds during test development
try:
    import maturin_import_hook
    maturin_import_hook.install()
except ImportError:
    pass  # Hook not available, rebuilds will be manual
