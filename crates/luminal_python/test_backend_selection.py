#!/usr/bin/env python3
"""
Manual verification script for backend selection.
Tests that LUMINAL_BACKEND environment variable correctly selects the backend.
"""

import os
import torch
from luminal import luminal_backend


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + 1.0


def test_backend(backend_name):
    """Test a specific backend"""
    os.environ["LUMINAL_BACKEND"] = backend_name
    print(f"\nTesting with LUMINAL_BACKEND={backend_name}")

    try:
        model = SimpleModel()
        compiled = torch.compile(model, backend=luminal_backend)

        # Create test input
        x = torch.randn(3, 3)

        # Run compiled model
        output = compiled(x)

        # Verify output is correct (x + 1)
        expected = x + 1.0
        if torch.allclose(output, expected, rtol=1e-4):
            print(f"✓ Backend '{backend_name}' works correctly!")
            print(f"  Input shape: {x.shape}")
            print(f"  Output matches expected: True")
            return True
        else:
            print(f"✗ Backend '{backend_name}' output mismatch!")
            return False
    except Exception as e:
        print(f"✗ Backend '{backend_name}' failed with error: {e}")
        return False


def test_invalid_backend():
    """Test that invalid backend raises an error"""
    print("\nTesting invalid backend value")
    os.environ["LUMINAL_BACKEND"] = "invalid"

    try:
        # Use a different model class to avoid cache
        class DifferentModel(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        model = DifferentModel()
        compiled = torch.compile(model, backend=luminal_backend)
        output = compiled(torch.randn(3, 3))
        print("✗ Invalid backend should have raised an error!")
        return False
    except ValueError as e:
        if "Invalid LUMINAL_BACKEND value: invalid" in str(e):
            print(f"✓ Invalid backend correctly raises ValueError")
            print(f"  Error message: {e}")
            return True
        else:
            print(f"✗ Unexpected error message: {e}")
            return False
    except Exception as e:
        # torch.compile wraps the error in BackendCompilerFailed
        if "Invalid LUMINAL_BACKEND value: invalid" in str(e):
            print(f"✓ Invalid backend correctly raises error (wrapped in {type(e).__name__})")
            return True
        print(f"✗ Unexpected exception type: {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("Backend Selection Verification Script")
    print("=" * 60)

    results = []

    # Test native backend (default)
    results.append(("native (default)", test_backend("native")))

    # Test case insensitivity
    results.append(("NATIVE (uppercase)", test_backend("NATIVE")))
    results.append(("Native (mixed case)", test_backend("Native")))

    # Test CUDA backend
    results.append(("cuda", test_backend("cuda")))
    results.append(("CUDA (uppercase)", test_backend("CUDA")))

    # Test invalid backend
    results.append(("invalid backend", test_invalid_backend()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
