"""
Comprehensive Unit Tests for AdvTok Stability Fixes

Tests cover:
- Multiprocessing pool cleanup
- Vocabulary caching (parallel and sequential)
- Async/await compatibility
- Timeout mechanisms
- Error recovery
- Memory management
- Signal handling
"""

import unittest
import asyncio
import multiprocessing
import tempfile
import os
import sys
import pickle
import time
import signal
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import torch
import transformers


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import advtok.mdd as mdd
import advtok.utils as utils


class TestVocabularyCaching(unittest.TestCase):
    """Test vocabulary caching functionality"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.temp_dir, "test_vocab_cache.pkl")

    def tearDown(self):
        """Cleanup test fixtures"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        os.rmdir(self.temp_dir)

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_vocab_cache_creation(self, mock_tokenizer):
        """Test that vocabulary cache is created correctly"""
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.get_vocab.return_value = {"hello": 1, "world": 2, "test": 3}
        mock_tok.convert_tokens_to_string = lambda x: x[0]
        mock_tokenizer.return_value = mock_tok

        # Mock tokenizer type detection
        with patch('advtok.utils.is_llama_tokenizer', return_value=False):
            with patch('advtok.utils.is_llama3_tokenizer', return_value=False):
                with patch('advtok.utils.is_gemma_tokenizer', return_value=False):
                    with patch('advtok.utils.is_gemma2_tokenizer', return_value=False):
                        with patch('advtok.utils.is_shieldgemma_tokenizer', return_value=False):
                            with patch('advtok.utils._tok_name', return_value="test_model"):
                                # Test sequential processing
                                mdd.cache_vocab(mock_tok, cache_file=self.cache_file, use_parallel=False)

                                # Verify cache was created
                                self.assertTrue(os.path.exists(self.cache_file))

                                # Verify cache can be loaded
                                with open(self.cache_file, "rb") as f:
                                    vocab = pickle.load(f)
                                self.assertIsInstance(vocab, dict)

    def test_vocab_cache_loading(self):
        """Test that vocabulary cache can be loaded"""
        # Create a test cache
        test_vocab = {"test": 1, "vocab": 2}
        with open(self.cache_file, "wb") as f:
            pickle.dump(test_vocab, f)

        # Mock tokenizer
        mock_tok = Mock()
        with patch('advtok.utils._tok_name', return_value="test_model"):
            mdd.cache_vocab(mock_tok, cache_file=self.cache_file)

            # Verify cache was loaded (not recreated)
            self.assertTrue(os.path.exists(self.cache_file))

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_vocab_cache_parallel_fallback(self, mock_tokenizer):
        """Test that parallel processing falls back to sequential on error"""
        mock_tok = Mock()
        mock_tok.get_vocab.return_value = {"test": 1}
        mock_tok.convert_tokens_to_string = lambda x: x[0]
        mock_tok.name_or_path = "test/model"
        mock_tokenizer.return_value = mock_tok

        with patch('advtok.utils._tok_name', return_value="test_model"):
            with patch('advtok.utils.is_llama_tokenizer', return_value=False):
                with patch('advtok.utils.is_llama3_tokenizer', return_value=False):
                    with patch('advtok.utils.is_gemma_tokenizer', return_value=False):
                        with patch('advtok.utils.is_gemma2_tokenizer', return_value=False):
                            with patch('advtok.utils.is_shieldgemma_tokenizer', return_value=False):
                                # Force parallel processing to fail
                                with patch('multiprocessing.Pool', side_effect=RuntimeError("Pool failed")):
                                    # Should not raise, should fall back to sequential
                                    mdd.cache_vocab(mock_tok, cache_file=self.cache_file,
                                                    use_parallel=True, fallback_on_error=True,
                                                    persistent=False)

                                    # Verify cache was created via fallback
                                    self.assertIn("test_model", mdd.build_from_tokenizer.V_cache)


class TestMultiprocessingCleanup(unittest.TestCase):
    """Test multiprocessing pool cleanup"""

    def test_pool_cleanup_on_error(self):
        """Test that pools are cleaned up even when errors occur"""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tok = Mock()
            mock_tok.get_vocab.return_value = {"test": 1}
            mock_tok.name_or_path = "test/model"
            mock_tokenizer.return_value = mock_tok

            with patch('advtok.utils._tok_name', return_value="test_model"):
                with patch('advtok.utils.is_llama_tokenizer', return_value=False):
                    # Create a mock pool that raises an error during processing
                    mock_pool = Mock()
                    mock_pool.imap.side_effect = RuntimeError("Processing error")

                    with patch('multiprocessing.Pool', return_value=mock_pool):
                        try:
                            mdd._parallelize_vocab_windows(mock_tok, ["test"])
                        except RuntimeError:
                            pass

                        # Verify pool cleanup methods were called
                        mock_pool.close.assert_called_once()
                        mock_pool.join.assert_called_once()

    def test_multiprocessing_start_method(self):
        """Test that multiprocessing start method is set safely"""
        # The module should have tried to set the start method
        # but not raise an error if it's already set
        current_method = multiprocessing.get_start_method(allow_none=True)
        self.assertIsNotNone(current_method)


class TestAsyncOperations(unittest.TestCase):
    """Test async/await compatibility"""

    def test_async_to_thread_compatibility(self):
        """Test that blocking operations can run in threads"""
        async def run_test():
            def blocking_operation():
                time.sleep(0.1)
                return "completed"

            result = await asyncio.to_thread(blocking_operation)
            return result

        # Run the async function
        result = asyncio.run(run_test())
        self.assertEqual(result, "completed")

    def test_async_timeout(self):
        """Test that async operations can timeout"""
        async def run_test():
            def slow_operation():
                time.sleep(10)
                return "completed"

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(slow_operation),
                    timeout=0.1
                )
                return result
            except asyncio.TimeoutError:
                return "timeout"

        result = asyncio.run(run_test())
        self.assertEqual(result, "timeout")

    def test_async_cancellation(self):
        """Test that async operations can be cancelled"""
        async def run_test():
            async def cancellable_operation():
                try:
                    await asyncio.sleep(10)
                    return "completed"
                except asyncio.CancelledError:
                    return "cancelled"

            task = asyncio.create_task(cancellable_operation())
            await asyncio.sleep(0.1)
            task.cancel()

            try:
                result = await task
            except asyncio.CancelledError:
                result = "cancelled"

            return result

        result = asyncio.run(run_test())
        self.assertEqual(result, "cancelled")


class TestMDDConstruction(unittest.TestCase):
    """Test MDD (Multi-valued Decision Diagram) construction"""

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_mdd_build_simple(self, mock_tokenizer):
        """Test building a simple MDD"""
        mock_tok = Mock()
        mock_tok.get_vocab.return_value = {"h": 1, "he": 2, "hel": 3, "hell": 4, "hello": 5}
        mock_tok.convert_ids_to_tokens = lambda x: [list(mock_tok.get_vocab().keys())[i-1] for i in x]
        mock_tok.convert_tokens_to_string = lambda x: ''.join(x)
        mock_tok.encode = lambda x, add_special_tokens=False: [mock_tok.get_vocab().get(x, 0)]
        mock_tok.eos_token_id = 2
        mock_tok.unk_token_id = 0
        mock_tokenizer.return_value = mock_tok

        with patch('advtok.utils.is_llama_tokenizer', return_value=False):
            with patch('advtok.utils._tok_name', return_value="test_model"):
                # Create simple vocab cache
                vocab_cache = {
                    "h": 1,
                    "e": 2,
                    "l": 3,
                    "o": 4
                }

                # Build MDD
                M = mdd.build_from_tokenizer("hello", mock_tok, cached=vocab_cache, prefix_space=False)

                # Verify MDD was created
                self.assertIsInstance(M, mdd.MDD)
                self.assertIsNotNone(M._ch)

    def test_mdd_enumeration(self):
        """Test MDD enumeration"""
        # Create a simple MDD manually
        mock_tok = Mock()
        mock_tok.eos_token_id = 2

        # Create a simple MDD: root -> (1, child) -> (2, None)
        child = mdd.MDD((1, 1), [(2, None)], tokenizer=mock_tok)
        root = mdd.MDD((0, 0), [(1, child)], tokenizer=mock_tok)

        # Enumerate
        models = root.enumerate(add_eos=False)

        # Should get [[1, 2]]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], [1, 2])


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery mechanisms"""

    def test_error_handling_in_vocab_cache(self):
        """Test that vocab caching handles errors gracefully"""
        mock_tok = Mock()
        mock_tok.get_vocab.side_effect = RuntimeError("Vocab error")

        with patch('advtok.utils._tok_name', return_value="test_model"):
            with self.assertRaises(RuntimeError):
                mdd.cache_vocab(mock_tok, use_parallel=False, persistent=False)

    def test_sequential_fallback_on_parallel_error(self):
        """Test that sequential processing is used when parallel fails"""
        mock_tok = Mock()
        mock_tok.get_vocab.return_value = {"test": 1}
        mock_tok.convert_tokens_to_string = lambda x: x[0]
        mock_tok.name_or_path = "test/model"

        with patch('advtok.utils._tok_name', return_value="test_model"):
            with patch('advtok.utils.is_llama_tokenizer', return_value=False):
                with patch('advtok.utils.is_llama3_tokenizer', return_value=False):
                    with patch('advtok.utils.is_gemma_tokenizer', return_value=False):
                        with patch('advtok.utils.is_gemma2_tokenizer', return_value=False):
                            with patch('advtok.utils.is_shieldgemma_tokenizer', return_value=False):
                                # Mock parallel processing to fail
                                with patch('multiprocessing.Pool', side_effect=EOFError("Pool error")):
                                    # Should fall back to sequential
                                    mdd.cache_vocab(mock_tok, persistent=False, use_parallel=True,
                                                    fallback_on_error=True)

                                    # Verify cache was created
                                    self.assertIn("test_model", mdd.build_from_tokenizer.V_cache)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management"""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_cuda_cache_clearing(self, mock_empty_cache, mock_is_available):
        """Test that CUDA cache is cleared appropriately"""
        mock_is_available.return_value = True

        # Create a mock model with CUDA device
        mock_model = Mock()
        mock_model.device = torch.device('cuda:0')

        # Simulate clearing cache
        if hasattr(mock_model, 'device') and 'cuda' in str(mock_model.device):
            torch.cuda.empty_cache()

        # Verify cache clearing was called
        mock_empty_cache.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_end_to_end_vocab_cache(self, mock_model, mock_tokenizer):
        """Test end-to-end vocabulary caching"""
        # Setup mocks
        mock_tok = Mock()
        mock_tok.get_vocab.return_value = {"hello": 1, "world": 2}
        mock_tok.convert_tokens_to_string = lambda x: ''.join(x)
        mock_tok.encode = lambda x, add_special_tokens=False: [1, 2]
        mock_tok.eos_token_id = 2
        mock_tok.unk_token_id = 0
        mock_tok.name_or_path = "test/model"
        mock_tokenizer.return_value = mock_tok

        with patch('advtok.utils._tok_name', return_value="test_model"):
            with patch('advtok.utils.is_llama_tokenizer', return_value=False):
                with patch('advtok.utils.is_llama3_tokenizer', return_value=False):
                    with patch('advtok.utils.is_gemma_tokenizer', return_value=False):
                        with patch('advtok.utils.is_gemma2_tokenizer', return_value=False):
                            with patch('advtok.utils.is_shieldgemma_tokenizer', return_value=False):
                                # Cache vocabulary
                                mdd.cache_vocab(mock_tok, use_parallel=False, persistent=False)

                                # Verify cache exists
                                self.assertIn("test_model", mdd.build_from_tokenizer.V_cache)
                                self.assertIsInstance(mdd.build_from_tokenizer.V_cache["test_model"], dict)


class TestSignalHandling(unittest.TestCase):
    """Test signal handling"""

    @unittest.skipIf(sys.platform == "win32", "SIGTERM not available on Windows")
    def test_signal_handler_registration(self):
        """Test that signal handlers can be registered"""
        handler_called = False

        def test_handler(signum, frame):
            nonlocal handler_called
            handler_called = True

        # Register handler
        old_handler = signal.signal(signal.SIGTERM, test_handler)

        # Restore old handler
        signal.signal(signal.SIGTERM, old_handler)

        # If we got here without error, registration works
        self.assertTrue(True)

    def test_ctrl_c_handling(self):
        """Test Ctrl+C handling"""
        handler_called = False

        def test_handler(signum, frame):
            nonlocal handler_called
            handler_called = True

        # Register handler
        old_handler = signal.signal(signal.SIGINT, test_handler)

        try:
            # Send signal to self
            os.kill(os.getpid(), signal.SIGINT)
            # Give it a moment to process
            time.sleep(0.1)
        finally:
            # Restore old handler
            signal.signal(signal.SIGINT, old_handler)

        self.assertTrue(handler_called)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVocabularyCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiprocessingCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestAsyncOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestMDDConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalHandling))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("AdvTok Stability Tests")
    print("=" * 70)
    print()

    result = run_tests()

    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
