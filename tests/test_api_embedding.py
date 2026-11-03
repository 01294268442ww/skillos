"""Offline SDK-mocked embedding behavior checks; no model or network calls."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
import numpy as np
import openai
from hyperskill.api_embedding import EmbeddingError, QwenEmbeddingClient


def config():
    # Three-dimensional vectors keep isolated tests small; no API is contacted.
    return {'EMBED_PROVIDER': 'compat', 'EMBED_MODEL': 'text-embedding-v3',
            'EMBED_BASE_URL': 'https://example.invalid/compatible-mode/v1',
            'EMBED_DIMENSIONS': '3', 'EMBED_API_KEY': 'offline-embedding-test-key',
            'DEEPSEEK_API_KEY': 'offline-llm-test-key', 'EMBED_TIMEOUT': '120',
            'EMBED_MAX_RETRIES': '40'}


def response(vectors=None, usage='total_only', model='text-embedding-v3'):
    vectors = [(0, [1.0, 0.0, 0.0])] if vectors is None else vectors
    if usage == 'total_only':
        usage = SimpleNamespace(total_tokens=7)
    elif usage == 'full':
        usage = SimpleNamespace(prompt_tokens=7, total_tokens=7)
    return SimpleNamespace(model=model, usage=usage,
        data=[SimpleNamespace(index=i, embedding=v) for i, v in vectors])


class StatusError(RuntimeError):
    def __init__(self, status):
        super().__init__('controlled offline failure; never serialize this message')
        self.status_code = status


class ApiEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.api = QwenEmbeddingClient(self.config, max_attempts=3, timeout=2)
        self.sleep = patch('hyperskill.api_embedding.time.sleep', return_value=None)
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()
        self.api.close()

    def test_exact_configuration_and_reordered_response(self):
        result = response([(1, [0.0, 1.0, 0.0]), (0, [1.0, 0.0, 0.0])])
        with patch.object(self.api.client.embeddings, 'create', return_value=result) as create:
            vectors = self.api.encode(['first', 'second'])
        np.testing.assert_allclose(vectors, [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(create.call_args.kwargs, {'model': 'text-embedding-v3',
            'input': ['first', 'second'], 'dimensions': 3, 'encoding_format': 'float'})
        self.assertEqual(self.api.profile['endpoint'], self.config['EMBED_BASE_URL'])
        public = json.dumps(self.api.public_configuration)
        self.assertNotIn(self.config['EMBED_API_KEY'], public)
        expected = hashlib.sha256(json.dumps(self.api.profile, sort_keys=True,
            ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(self.api.profile_id, expected)

    def test_missing_configuration_never_chooses_another_model_or_endpoint(self):
        for key in ('EMBED_MODEL', 'EMBED_BASE_URL', 'EMBED_DIMENSIONS', 'EMBED_API_KEY', 'EMBED_PROVIDER'):
            with self.subTest(key=key):
                data = config();data.pop(key)
                with patch('hyperskill.api_embedding.openai.OpenAI') as constructor:
                    with self.assertRaises(EmbeddingError):
                        QwenEmbeddingClient(data)
                    constructor.assert_not_called()

    def test_empty_input_is_not_an_api_request(self):
        with patch.object(self.api.client.embeddings, 'create') as create:
            matrix = self.api.encode([])
        self.assertEqual(matrix.shape, (0, 3))
        create.assert_not_called()
        self.assertEqual(self.api.usage_summary()['request_attempts'], 0)
        self.assertIsNone(self.api.usage_summary()['total_usage'])

    def test_bad_inputs_and_actual_config_secrets_are_refused(self):
        for texts in ('plain string', [''], [True], ['\ud800'],
                      ['prefix '+self.config['EMBED_API_KEY']], [self.config['DEEPSEEK_API_KEY']]):
            with self.subTest(texts_type=type(texts).__name__):
                with patch.object(self.api.client.embeddings, 'create') as create:
                    with self.assertRaises(EmbeddingError) as failed:
                        self.api.encode(texts)
                    create.assert_not_called()
                    self.assertNotIn(self.config['EMBED_API_KEY'], str(failed.exception))

    def test_v3_batch_size_limit_and_order_are_preserved(self):
        texts = ['text '+str(i) for i in range(21)]
        def generate(**kwargs):
            data = kwargs['input']
            return response([(i, [float(int(t.split()[-1])+1), 1.0, 0.0]) for i, t in enumerate(data)])
        with patch.object(self.api.client.embeddings, 'create', side_effect=generate) as create:
            matrix = self.api.encode(texts)
        self.assertEqual([len(x.kwargs['input']) for x in create.call_args_list], [10, 10, 1])
        np.testing.assert_allclose(matrix[:, 0], list(range(1, 22)))
        self.assertEqual([x['input_offset'] for x in self.api.call_ledger], [0, 10, 20])
        self.assertEqual(self.api.usage_summary()['total_usage']['total_tokens'], 21)

    def test_bad_dimensions_indices_and_numbers_never_produce_fake_vectors(self):
        malformed = [
            response([(0, [1.0, 0.0])]), response([(True, [1.0, 0.0, 0.0])]),
            response([(1, [1.0, 0.0, 0.0])]), response([(0, [float('nan'), 0.0, 1.0])]),
            response([(0, [float('inf'), 0.0, 1.0])]), response([(0, [True, 0.0, 1.0])]),
            response([(0, [0.0, 0.0, 0.0])]), response([]),
        ]
        for value in malformed:
            with self.subTest(kind=repr(value.data)):
                with patch.object(self.api.client.embeddings, 'create', return_value=value) as create:
                    with self.assertRaises(EmbeddingError):
                        self.api.encode(['test'])
                self.assertEqual(create.call_count, 1)
                self.assertEqual(self.api.call_ledger[-1]['outcome'], 'invalid_response')
        # Invalid vectors were still billed responses; their valid provider cost remains.
        self.assertEqual(self.api.usage_summary()['total_usage']['total_tokens'], 7*len(malformed))

    def test_duplicate_response_indices_are_rejected(self):
        duplicate = response([(0, [1.0, 0.0, 0.0]), (0, [0.0, 1.0, 0.0])])
        with patch.object(self.api.client.embeddings, 'create', return_value=duplicate):
            with self.assertRaises(EmbeddingError) as failed:
                self.api.encode(['first', 'second'])
        self.assertEqual(failed.exception.category, 'invalid_embedding_response_indices')

    def test_network_or_status_failure_is_bounded_without_sdk_hidden_retries(self):
        request = self.api.client._client.build_request('POST', 'https://example.invalid/v1/embeddings')
        error = openai.APIConnectionError(request=request)
        with patch.object(self.api.client.embeddings, 'create', side_effect=error) as create:
            with self.assertRaises(EmbeddingError):
                self.api.encode(['test'])
        self.assertEqual(create.call_count, 3)
        self.assertEqual(self.api.usage_summary()['unknown_usage_attempts'], [1, 2, 3])
        self.assertIsNone(self.api.usage_summary()['total_usage'])
        self.assertEqual(self.api.public_configuration['sdk_retries'], 0)
        self.assertNotIn('controlled offline failure', json.dumps(self.api.call_ledger))

    def test_later_batch_failure_keeps_first_batch_billed_usage(self):
        first = response([(i, [1.0, 0.0, 0.0]) for i in range(10)])
        with patch.object(self.api.client.embeddings, 'create', side_effect=[first, StatusError(401)]) as create:
            with self.assertRaises(EmbeddingError):
                self.api.encode(['input '+str(i) for i in range(11)])
        self.assertEqual(create.call_count, 2)
        summary = self.api.usage_summary()
        self.assertEqual(summary['known_usage']['total_tokens'], 7)
        self.assertIsNone(summary['total_usage'])
        self.assertEqual(summary['unknown_usage_attempts'], [2])

    def test_rate_limit_retries_never_exceed_three_total_attempts(self):
        with QwenEmbeddingClient(config(), max_attempts=99, timeout=1) as limited:
            with patch.object(limited.client.embeddings, 'create', side_effect=StatusError(429)) as create:
                with self.assertRaises(EmbeddingError):limited.encode(['test'])
            self.assertEqual(create.call_count, 3)
            self.assertEqual(limited.effective_attempts, 3)
            self.assertTrue(all(row['http_status']==429 for row in limited.call_ledger))

    def test_total_only_usage_is_known_without_fabricated_prompt_tokens(self):
        with patch.object(self.api.client.embeddings, 'create', return_value=response()):
            self.api.encode(['test'])
        summary = self.api.usage_summary()
        self.assertTrue(summary['usage_complete'])
        self.assertEqual(summary['total_usage'], {'total_tokens': 7, 'prompt_tokens': None})
        self.assertFalse(summary['prompt_usage_complete'])
        self.assertEqual(summary['unknown_prompt_usage_attempts'], [1])
        self.assertNotIn('prompt_tokens', self.api.call_ledger[0]['usage'])

    def test_unknown_and_invalid_total_usage_are_distinct(self):
        values = [response(usage=None), response(usage=SimpleNamespace(total_tokens=True)),
                  response(usage='full')]
        with patch.object(self.api.client.embeddings, 'create', side_effect=values):
            for _ in values:self.api.encode(['test'])
        summary = self.api.usage_summary()
        self.assertFalse(summary['usage_complete'])
        self.assertIsNone(summary['total_usage'])
        self.assertEqual(summary['unknown_usage_attempts'], [1])
        self.assertEqual(summary['invalid_usage_attempts'], [2])
        self.assertEqual(summary['known_usage'], {'prompt_tokens': 7, 'total_tokens': 7})

    def test_response_model_mismatch_does_not_change_configured_model(self):
        with patch.object(self.api.client.embeddings, 'create', return_value=response(model='another-model')):
            with self.assertRaises(EmbeddingError) as failed:
                self.api.encode(['test'])
        self.assertEqual(failed.exception.category, 'embedding_response_model_mismatch')
        self.assertEqual(self.api.model, 'text-embedding-v3')
        self.assertEqual(self.api.usage_summary()['total_usage']['total_tokens'], 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
