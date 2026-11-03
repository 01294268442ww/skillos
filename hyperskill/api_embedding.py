"""Audited Qwen embedding API adapter; no local-model or endpoint fallback."""
import hashlib
import json
import math
import re
import time
from urllib.parse import urlsplit

import numpy as np
import openai


class EmbeddingError(RuntimeError):
    """A controlled category; never include raw SDK responses or credentials."""
    def __init__(self, category):
        self.category = category if isinstance(category, str) and re.fullmatch(
            r"[a-z][a-z0-9_]{0,79}", category) else "embedding_error"
        super().__init__(self.category)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive_number(value):
    if isinstance(value, bool):
        raise EmbeddingError("invalid_embedding_execution_limit")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise EmbeddingError("invalid_embedding_execution_limit") from None
    if not math.isfinite(number) or number <= 0:
        raise EmbeddingError("invalid_embedding_execution_limit")
    return number


def _positive_integer(value, category):
    if isinstance(value, bool):
        raise EmbeddingError(category)
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]+", value):
            raise EmbeddingError(category)
        value = int(value)
    if type(value) is not int or value < 1:
        raise EmbeddingError(category)
    return value


class QwenEmbeddingClient:
    """Expose SentenceTransformer's list encode interface over the configured API."""
    def __init__(self, config, max_attempts=3, timeout=60):
        if (not isinstance(config, dict) or any(not isinstance(k, str) for k in config)
                or config.get("EMBED_PROVIDER") != "compat"):
            raise EmbeddingError("embedding_provider_must_be_explicit_compat")
        required = ("EMBED_MODEL", "EMBED_BASE_URL", "EMBED_API_KEY")
        if (any(not isinstance(config.get(k), str) or not config[k].strip() for k in required)
                or config.get("EMBED_DIMENSIONS") is None):
            raise EmbeddingError("required_embedding_configuration_missing")
        self.model = config["EMBED_MODEL"]
        self.endpoint = config["EMBED_BASE_URL"]
        self.dimensions = _positive_integer(config["EMBED_DIMENSIONS"], "invalid_embedding_dimensions")
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,119}", self.model):
            raise EmbeddingError("embedding_model_must_be_explicit_literal")
        if not isinstance(self.endpoint, str):
            raise EmbeddingError("invalid_embedding_endpoint")
        try:
            endpoint = urlsplit(self.endpoint)
            valid = (endpoint.scheme == "https" and endpoint.hostname and not endpoint.username
                     and not endpoint.password and not endpoint.query and not endpoint.fragment)
            endpoint.port
        except ValueError:
            valid = False
        if not valid:
            raise EmbeddingError("invalid_embedding_endpoint")
        self._secrets = tuple(v for k, v in config.items() if isinstance(v, str) and v and (
            "API_KEY" in k.upper() or k.lower() in ("token", "password", "secret")
            or k.lower().endswith(("_access_token", "_refresh_token", "_password", "_secret"))))
        if any(secret in self.model or secret in self.endpoint for secret in self._secrets):
            raise EmbeddingError("secret_value_not_allowed_in_embedding_profile")
        attempts = _positive_integer(max_attempts, "invalid_embedding_attempt_limit")
        configured_attempts = _positive_integer(config.get("EMBED_MAX_RETRIES", attempts), "invalid_embedding_attempt_limit")
        self.effective_attempts = min(3, attempts, configured_attempts)
        # The configured v3 API permits at most 10 strings in one request.
        self.batch_size = min(10, _positive_integer(config.get("EMBED_BATCH_SIZE", 10), "invalid_embedding_batch_size"))
        self.effective_timeout = min(_positive_number(timeout), _positive_number(config.get("EMBED_TIMEOUT", timeout)))
        verify = str(config.get("EMBED_SSL_VERIFY", "1")).lower()
        if verify not in ("0", "1", "true", "false", "yes", "no", "on", "off"):
            raise EmbeddingError("invalid_embedding_ssl_verify")
        self.ssl_verify = verify in ("1", "true", "yes", "on")
        self.profile = {"provider": "compat", "model": self.model, "endpoint": self.endpoint,
                        "dimensions": self.dimensions, "encoding_format": "float", "vector_dtype": "float32"}
        self.profile_id = _fingerprint(self.profile)
        self.public_configuration = {**self.profile, "profile_id": self.profile_id,
                                     "effective_timeout_seconds": self.effective_timeout,
                                     "max_total_attempts": self.effective_attempts,
                                     "batch_size": self.batch_size,
                                     "sdk_retries": 0, "ssl_verify": self.ssl_verify}
        self.call_ledger = []
        self.encode_count = 0
        self.client = openai.OpenAI(
            base_url=self.endpoint, api_key=config["EMBED_API_KEY"],
            timeout=self.effective_timeout, max_retries=0,
            http_client=openai.DefaultHttpxClient(verify=self.ssl_verify))

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _vectors(self, response, count):
        rows = getattr(response, "data", None)
        if not isinstance(rows, (list, tuple)) or len(rows) != count:
            raise EmbeddingError("embedding_response_count_mismatch")
        ordered = [None] * count
        for row in rows:
            index, vector = getattr(row, "index", None), getattr(row, "embedding", None)
            if type(index) is not int or not 0 <= index < count or ordered[index] is not None:
                raise EmbeddingError("invalid_embedding_response_indices")
            if not isinstance(vector, (list, tuple)) or len(vector) != self.dimensions:
                raise EmbeddingError("embedding_response_dimension_mismatch")
            try:
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                    raise EmbeddingError("invalid_embedding_vector")
                with np.errstate(over="ignore", invalid="ignore"):
                    converted = np.asarray(vector, dtype=np.float32)
                if not np.isfinite(converted).all() or not np.any(converted != 0):
                    raise EmbeddingError("invalid_embedding_vector")
            except (OverflowError, TypeError, ValueError):
                raise EmbeddingError("invalid_embedding_vector") from None
            ordered[index] = converted
        return np.stack(ordered)

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        if (not isinstance(texts, (list, tuple))
                or any(not isinstance(t, str) or not t.strip() for t in texts)):
            raise EmbeddingError("embedding_input_must_be_nonempty_text_list")
        try:
            hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        except UnicodeError:
            raise EmbeddingError("invalid_embedding_input_unicode") from None
        if any(secret in text for text in texts for secret in self._secrets):
            raise EmbeddingError("secret_value_not_allowed_in_embedding_input")
        if not texts:
            matrix = np.empty((0, self.dimensions), dtype=np.float32)
            return matrix if convert_to_numpy else matrix.tolist()
        self.encode_count += 1
        batches = []
        for offset in range(0, len(texts), self.batch_size):
            batches.append(self._encode_batch(list(texts[offset:offset+self.batch_size]),
                hashes[offset:offset+self.batch_size], offset))
        matrix = np.concatenate(batches, axis=0)
        return matrix if convert_to_numpy else matrix.tolist()

    def _encode_batch(self, texts, hashes, offset):
        for attempt in range(1, self.effective_attempts + 1):
            started = time.monotonic()
            row = {"sequence": len(self.call_ledger)+1, "phase": "embedding", "attempt": attempt,
                   "requested_model": self.model, "profile_id": self.profile_id,
                   "input_count": len(texts), "input_hashes": hashes,
                   "encode_sequence": self.encode_count, "input_offset": offset,
                   "usage": None, "outcome": "request_failed"}
            self.call_ledger.append(row)
            try:
                response = self.client.embeddings.create(model=self.model, input=list(texts),
                    dimensions=self.dimensions, encoding_format="float")
            except Exception as error:
                row.update(error_type=type(error).__name__, category="embedding_request_failed",
                           duration_seconds=round(time.monotonic()-started, 6))
                status = getattr(error, "status_code", None)
                if type(status) is int:
                    row["http_status"] = status
                retryable = isinstance(error, openai.APIConnectionError) or status == 429 or (
                    type(status) is int and status >= 500)
                if retryable and attempt < self.effective_attempts:
                    time.sleep(min(2 ** (attempt-1), 4))
                    continue
                raise EmbeddingError("embedding_request_failed") from None
            row.update(http_status=200, outcome="response_received",
                       duration_seconds=round(time.monotonic()-started, 6))
            usage = getattr(response, "usage", None)
            if usage is not None:
                row["usage"] = {}
                for key in ("prompt_tokens", "total_tokens"):
                    value = getattr(usage, key, None)
                    if value is not None:
                        row["usage"][key] = value if type(value) in (int, bool) else None
                        if type(value) is not int or value < 0:
                            row.setdefault("invalid_usage_fields", []).append(key)
            response_model = getattr(response, "model", None)
            if (isinstance(response_model, str) and re.fullmatch(r"[A-Za-z0-9_./:-]{1,120}", response_model)
                    and not any(secret in response_model for secret in self._secrets)):
                row["response_model"] = response_model
            try:
                if response_model is not None and response_model != self.model:
                    raise EmbeddingError("embedding_response_model_mismatch")
                matrix = self._vectors(response, len(texts))
            except EmbeddingError as error:
                row.update(outcome="invalid_response", category=error.category)
                raise
            return matrix
        raise EmbeddingError("embedding_attempt_budget_exhausted")

    def usage_summary(self):
        def valid_total(row):
            usage = row.get("usage")
            return (isinstance(usage, dict) and type(usage.get("total_tokens")) is int
                    and usage["total_tokens"] >= 0)
        def valid_prompt(row):
            usage = row.get("usage")
            return (isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int
                    and usage["prompt_tokens"] >= 0 and
                    (not valid_total(row) or usage["prompt_tokens"] <= usage["total_tokens"]))
        rows = [r for r in self.call_ledger if valid_total(r)]
        prompt_rows = [r for r in self.call_ledger if valid_prompt(r)]
        complete = bool(self.call_ledger) and len(rows) == len(self.call_ledger)
        prompt_complete = bool(self.call_ledger) and len(prompt_rows) == len(self.call_ledger)
        known = {"total_tokens": sum(r["usage"]["total_tokens"] for r in rows),
                 "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in prompt_rows)}
        total = {"total_tokens": known["total_tokens"],
                 "prompt_tokens": known["prompt_tokens"] if prompt_complete else None}
        def missing(row, key):
            return ((row["usage"] is None or row["usage"].get(key) is None)
                    and key not in row.get("invalid_usage_fields", []))
        return {"request_attempts": len(self.call_ledger), "usage_complete": complete,
                "unknown_usage_attempts": [r["sequence"] for r in self.call_ledger if missing(r, "total_tokens")],
                "invalid_usage_attempts": [r["sequence"] for r in self.call_ledger if not missing(r, "total_tokens") and not valid_total(r)],
                "prompt_usage_complete": prompt_complete,
                "unknown_prompt_usage_attempts": [r["sequence"] for r in self.call_ledger if missing(r, "prompt_tokens")],
                "invalid_prompt_usage_attempts": [r["sequence"] for r in self.call_ledger if not missing(r, "prompt_tokens") and not valid_prompt(r)],
                "known_usage": known, "total_usage": total if complete else None}
