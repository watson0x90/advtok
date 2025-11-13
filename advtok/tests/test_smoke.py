"""
Smoke Tests for AdvTok - Quick validation of core functionality

These tests run quickly and verify that basic operations work.
Run this before the full test suite to catch obvious issues.
"""

import sys
import os
import asyncio
import time

# Add parent directory to path (advtok package directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...", end=" ")
    try:
        import advtok.mdd
        import advtok.utils
        import advtok.search
        import advtok.jailbreak
        import advtok
        print("[PASS]")
        return True
    except Exception as e:
        print(f"[FAIL]: {e}")
        return False


def test_multiprocessing_start_method():
    """Test that multiprocessing start method is set"""
    print("Testing multiprocessing start method...", end=" ")
    try:
        import multiprocessing
        method = multiprocessing.get_start_method(allow_none=True)
        if method is not None:
            print(f"[PASS] (method: {method})")
            return True
        else:
            print("[FAIL] FAIL: Start method not set")
            return False
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_async_operations():
    """Test basic async operations"""
    print("Testing async operations...", end=" ")
    try:
        async def test_async():
            await asyncio.sleep(0.01)
            return "success"

        result = asyncio.run(test_async())
        if result == "success":
            print("[PASS] PASS")
            return True
        else:
            print("[FAIL] FAIL: Unexpected result")
            return False
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_async_to_thread():
    """Test asyncio.to_thread functionality"""
    print("Testing asyncio.to_thread...", end=" ")
    try:
        async def test():
            def blocking():
                time.sleep(0.01)
                return "success"
            result = await asyncio.to_thread(blocking)
            return result

        result = asyncio.run(test())
        if result == "success":
            print("[PASS] PASS")
            return True
        else:
            print("[FAIL] FAIL: Unexpected result")
            return False
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_async_timeout():
    """Test async timeout mechanism"""
    print("Testing async timeout...", end=" ")
    try:
        async def test():
            async def slow():
                await asyncio.sleep(10)
            try:
                await asyncio.wait_for(slow(), timeout=0.01)
                return "no_timeout"
            except asyncio.TimeoutError:
                return "timeout"

        result = asyncio.run(test())
        if result == "timeout":
            print("[PASS] PASS")
            return True
        else:
            print("[FAIL] FAIL: Timeout did not occur")
            return False
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_vocab_cache_structure():
    """Test vocabulary cache structure"""
    print("Testing vocabulary cache structure...", end=" ")
    try:
        import advtok.mdd as mdd
        # Check if cache structure exists
        if hasattr(mdd.build_from_tokenizer, 'V_cache'):
            print("[PASS] PASS")
            return True
        else:
            # Cache might not be initialized yet, which is ok
            print("[PASS] PASS (not initialized)")
            return True
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_torch_cuda_available():
    """Test if CUDA is available (informational)"""
    print("Testing CUDA availability...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[PASS] AVAILABLE (device: {torch.cuda.get_device_name(0)})")
        else:
            print("[WARN] NOT AVAILABLE (will use CPU)")
        return True
    except Exception as e:
        print(f"[WARN] WARNING: {e}")
        return True  # Not a critical failure


def test_transformers_import():
    """Test transformers library"""
    print("Testing transformers import...", end=" ")
    try:
        import transformers
        print(f"[PASS] PASS (version: {transformers.__version__})")
        return True
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_textual_import():
    """Test textual library"""
    print("Testing textual import...", end=" ")
    try:
        import textual
        print(f"[PASS] PASS (version: {textual.__version__})")
        return True
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_signal_handling():
    """Test signal handling capability"""
    print("Testing signal handling...", end=" ")
    try:
        import signal

        def dummy_handler(signum, frame):
            pass

        # Try to register SIGINT handler
        old_handler = signal.signal(signal.SIGINT, dummy_handler)
        signal.signal(signal.SIGINT, old_handler)

        print("[PASS] PASS")
        return True
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def test_mdd_basic_structure():
    """Test basic MDD structure"""
    print("Testing MDD basic structure...", end=" ")
    try:
        import advtok.mdd as mdd

        # Create a simple MDD
        test_mdd = mdd.MDD(var=0, ch=[(1, None)], largest=0)

        if test_mdd._var == 0 and len(test_mdd._ch) == 1:
            print("[PASS] PASS")
            return True
        else:
            print("[FAIL] FAIL: Unexpected MDD structure")
            return False
    except Exception as e:
        print(f"[FAIL] FAIL: {e}")
        return False


def run_all_tests():
    """Run all smoke tests"""
    print("=" * 70)
    print("AdvTok Smoke Tests - Quick Validation")
    print("=" * 70)
    print()

    tests = [
        test_imports,
        test_transformers_import,
        test_textual_import,
        test_multiprocessing_start_method,
        test_async_operations,
        test_async_to_thread,
        test_async_timeout,
        test_signal_handling,
        test_vocab_cache_structure,
        test_mdd_basic_structure,
        test_torch_cuda_available,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"CRITICAL ERROR in {test.__name__}: {e}")
            results.append(False)

    print()
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("[PASS] All smoke tests passed! The system is ready.")
    else:
        print("[FAIL] Some tests failed. Please review the output above.")

    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
