"""Bounded, auditable DeepSeek requests for the native no-skill agent.

This is a functional smoke protocol, not the final experimental B definition.
Credentials are read from one explicit dotenv file and never serialized.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlsplit

from dotenv import dotenv_values
import openai

from FlashOAgents.models import (
    ChatMessage, OpenAIServerModel, parse_tool_args_if_needed,
)


class SafeRuntimeError(RuntimeError):
    """Contains a fixed error category, never an SDK response or credential."""

    def __init__(self, category):
        self.category = category if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", category) else "invalid_error_category"
        super().__init__(self.category)


def safe_error(error):
    result = {"type": type(error).__name__}
    if isinstance(error, SafeRuntimeError):
        result["category"] = error.category
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    return result


def load_runtime_config(path):
    path = Path(path).expanduser()
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise SafeRuntimeError("runtime_config_must_be_private_regular_file")
    values = {k: v for k, v in dotenv_values(path, interpolate=False).items()
              if v is not None}
    for key in ("MODEL", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"):
        if not values.get(key):
            raise SafeRuntimeError("required_runtime_configuration_missing")
    if any(char in values["MODEL"] for char in ("$", "`", "\n", "\r")):
        raise SafeRuntimeError("model_must_be_explicit_literal")
    endpoint = urlsplit(values["DEEPSEEK_BASE_URL"])
    if (endpoint.scheme not in ("https", "http") or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise SafeRuntimeError("invalid_runtime_endpoint")
    return values


def apply_tool_environment(config):
    # Official modules call load_dotenv at import time. Apply the explicit
    # configuration after those imports; unrelated .env files do not select tools.
    os.environ["AZURE_OPENAI"] = "0"
    os.environ["WEB_ACCESS_PROVIDER"] = config.get("WEB_ACCESS_PROVIDER", "direct")
    os.environ["WEB_SEARCH_PROVIDER"] = config.get("WEB_SEARCH_PROVIDER", "auto")
    for key in ("SERPER_API_KEY", "JINA_API_KEY"):
        if config.get(key):
            os.environ[key] = config[key]
        else:
            os.environ.pop(key, None)


def thinking_extra_body(config):
    # Exact Agentless presets and shallow custom-body override semantics.
    presets = {
        "none": {},
        "thinking_type": {"thinking": {"type": "disabled"}},
        "enable_thinking": {"enable_thinking": False},
        "chat_template_kwargs": {
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}},
    }
    presets["all"] = {**presets["thinking_type"], **presets["enable_thinking"],
                      **presets["chat_template_kwargs"]}
    preset = config.get("DEEPSEEK_DISABLE_THINKING", "all").strip() or "all"
    if preset not in presets:
        raise SafeRuntimeError("unsupported_thinking_preset")
    extra = json.loads(json.dumps(presets[preset]))
    custom = config.get("DEEPSEEK_EXTRA_BODY", "").strip()
    if custom:
        try:
            custom = json.loads(custom)
        except (ValueError, TypeError):
            raise SafeRuntimeError("invalid_extra_body_json") from None
        if not isinstance(custom, dict):
            raise SafeRuntimeError("extra_body_must_be_object")
        extra.update(custom)
    return extra


def redact(value, secrets):
    """Whitelist serialization; SDK/raw objects cannot reach a report."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, ChatMessage):
        return redact({"role": value.role, "content": value.content,
                       "reasoning_content": value.reasoning_content,
                       "tool_calls": value.tool_calls}, secrets)
    if isinstance(value, (list, tuple)):
        return [redact(x, secrets) for x in value]
    if isinstance(value, dict):
        return {redact(str(k), secrets): redact(v, secrets) for k, v in value.items()}
    # No arbitrary str()/__dict__ fallback: those may include raw HTTP objects.
    return {"unserialized_type": type(value).__name__}


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def native_final_answer_evidence(steps):
    """Inspect native termination evidence; a max-step fallback is not success."""
    calls, errors = [], []
    for step in steps:
        error = getattr(step, "error", None)
        if error is not None:
            errors.append({"step_number": getattr(step, "step_number", None),
                           "type": type(error).__name__})
        for call in getattr(step, "tool_calls", None) or []:
            if call.name != "final_answer":
                continue
            arguments = call.arguments
            answer = arguments.get("answer") if isinstance(arguments, dict) else arguments
            present = answer is not None
            if isinstance(answer, str):
                present = bool(answer.strip())
            elif isinstance(answer, (dict, list, tuple)):
                present = bool(answer)
            calls.append({"step_number": getattr(step, "step_number", None),
                          "nonempty_answer": present})
    return {"final_answer_calls": calls, "step_errors": errors,
            "called": bool(calls), "nonempty_answer": any(c["nonempty_answer"] for c in calls),
            "max_steps_fallback": any(e["type"] == "AgentMaxStepsError" for e in errors)}


def _boolean(config, key, default=True):
    raw = config.get(key)
    if raw is None:
        return default
    if raw.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if raw.strip().lower() in ("0", "false", "no", "off"):
        return False
    raise SafeRuntimeError("invalid_boolean_configuration")


class AuditedOpenAIServerModel(OpenAIServerModel):
    def __init__(self, config, max_tokens=2048, max_api_attempts=3, request_timeout=90):
        if max_tokens < 1 or max_api_attempts < 1 or request_timeout <= 0:
            raise SafeRuntimeError("invalid_smoke_execution_limit")
        self.configured_timeout = float(config.get("DEEPSEEK_TIMEOUT", 300))
        self.configured_attempts = int(config.get("DEEPSEEK_MAX_RETRIES", 40))
        if self.configured_timeout <= 0 or self.configured_attempts < 1:
            raise SafeRuntimeError("invalid_configured_execution_limit")
        self.effective_timeout = min(self.configured_timeout, request_timeout)
        self.effective_attempts = min(self.configured_attempts, max_api_attempts)
        self.extra_body = thinking_extra_body(config)
        self.strip_think = _boolean(config, "DEEPSEEK_STRIP_THINK")
        self.ssl_verify = _boolean(config, "DEEPSEEK_SSL_VERIFY")
        super().__init__(
            model_id=config["MODEL"], api_base=config["DEEPSEEK_BASE_URL"],
            api_key=config["DEEPSEEK_API_KEY"], temperature=0.0,
            max_tokens=max_tokens,
            custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
        )
        self.client.close()
        self.client = openai.OpenAI(
            base_url=config["DEEPSEEK_BASE_URL"], api_key=config["DEEPSEEK_API_KEY"],
            timeout=self.effective_timeout, max_retries=0,
            http_client=openai.DefaultHttpxClient(verify=self.ssl_verify),
        )
        self.call_ledger = []
        self.phase = "agent"
        secret_values = [v for k, v in config.items() if v and any(
            marker in k.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))]
        self.public_configuration = {
            "model": config["MODEL"], "endpoint": config["DEEPSEEK_BASE_URL"],
            "temperature": 0.0, "max_tokens": max_tokens,
            "ssl_verify": self.ssl_verify, "strip_think": self.strip_think,
            "extra_body": redact(self.extra_body, secret_values),
            "configured": {"timeout_seconds": self.configured_timeout,
                           "max_total_attempts": self.configured_attempts},
            "smoke_limits": {"timeout_seconds": request_timeout,
                             "max_total_attempts": max_api_attempts},
            "effective": {"timeout_seconds": self.effective_timeout,
                          "max_total_attempts": self.effective_attempts,
                          "sdk_retries": 0},
        }

    def __call__(self, messages, stop_sequences=None, grammar=None,
                 tools_to_call_from=None, **kwargs):
        request = self._prepare_completion_kwargs(
            messages=messages, stop_sequences=stop_sequences, grammar=grammar,
            tools_to_call_from=tools_to_call_from, model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True, **kwargs,
        )
        if self.extra_body:
            request["extra_body"] = {**request.get("extra_body", {}), **self.extra_body}
        if any(name in self.model_id.lower() for name in ("o3", "o4")):
            request.pop("stop", None)
        for attempt in range(1, self.effective_attempts + 1):
            started = time.monotonic()
            row = {"sequence": len(self.call_ledger) + 1, "phase": self.phase,
                   "attempt": attempt, "requested_model": self.model_id,
                   "request_max_tokens": request.get("max_tokens"),
                   "usage": None, "outcome": "request_failed"}
            self.call_ledger.append(row)
            try:
                response = self.client.chat.completions.create(**request)
            except Exception as error:
                row.update(safe_error(error))
                row["duration_seconds"] = round(time.monotonic() - started, 6)
                status = row.get("http_status")
                retryable = isinstance(error, openai.APIConnectionError) or status == 429 or (
                    isinstance(status, int) and status >= 500)
                if retryable and attempt < self.effective_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise SafeRuntimeError("llm_request_failed") from None
            row.update({"http_status": 200, "response_id": response.id,
                        "response_model": response.model, "outcome": "response_received",
                        "duration_seconds": round(time.monotonic() - started, 6)})
            if response.usage is not None:
                row["usage"] = response.usage.model_dump(exclude_none=True)
                incoming = response.usage.prompt_tokens
                outgoing = response.usage.completion_tokens
                if isinstance(incoming, int) and isinstance(outgoing, int):
                    self.last_input_token_count = incoming
                    self.last_output_token_count = outgoing
                    self.total_input_tokens += incoming
                    self.total_output_tokens += outgoing
            self.total_api_calls += 1
            if not response.choices:
                row["outcome"] = "empty_choices"
                raise SafeRuntimeError("llm_response_has_no_choices")
            choice = response.choices[0]
            row["finish_reason"] = choice.finish_reason
            if not choice.message.content and not choice.message.tool_calls:
                row["outcome"] = "empty_content"
                if attempt < self.effective_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise SafeRuntimeError("llm_response_has_no_content")
            message = ChatMessage.from_dict(choice.message.model_dump(
                include={"role", "content", "tool_calls", "reasoning_content"}))
            message.raw = response
            if message.content and self.strip_think:
                message.content = self.remove_think_tags(message.content)
            if message.content and any(x in self.model_id.lower() for x in ("o3", "o4")):
                message.content = self.truncate_content_based_on_stop_sequences(
                    message.content, stop_sequences)
            return parse_tool_args_if_needed(message) if tools_to_call_from is not None else message
        raise SafeRuntimeError("llm_attempt_budget_exhausted")

    def usage_summary(self):
        def valid_usage(row):
            usage = row.get("usage")
            return (isinstance(usage, dict)
                    and all(type(usage.get(k)) is int and usage[k] >= 0
                            for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"])
        valid_rows = [x for x in self.call_ledger if valid_usage(x)]
        complete = bool(self.call_ledger) and len(valid_rows) == len(self.call_ledger)
        known = {k: sum(x["usage"][k] for x in valid_rows)
                 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return {"request_attempts": len(self.call_ledger), "usage_complete": complete,
                "unknown_usage_attempts": [x["sequence"] for x in self.call_ledger if x.get("usage") is None],
                "invalid_usage_attempts": [x["sequence"] for x in self.call_ledger
                                           if x.get("usage") is not None and not valid_usage(x)],
                "known_usage": known,
                "total_usage": known if complete else None}
