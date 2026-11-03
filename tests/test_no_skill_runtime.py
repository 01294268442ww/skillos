"""Offline behavior checks. Run with HYPERSKILL_REPO pointing at the checkout."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

repo = Path(os.environ.get("HYPERSKILL_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(repo))
import openai
from openai.types.chat import ChatCompletionMessage
from FlashOAgents.memory import ActionStep, ToolCall
from FlashOAgents.monitoring import AgentLogger, LogLevel
from FlashOAgents.utils import AgentMaxStepsError
from hyperskill.no_skill_runtime import (
    AuditedOpenAIServerModel, SafeRuntimeError, native_final_answer_evidence, safe_error,
)


def response(content, incoming=8, outgoing=2, total=10):
    fields = {"prompt_tokens": incoming, "completion_tokens": outgoing, "total_tokens": total}
    usage = SimpleNamespace(**fields, model_dump=lambda **kwargs: dict(fields))
    return SimpleNamespace(
        id="offline-response", model="deepseek-v4.1-flash", usage=usage,
        choices=[SimpleNamespace(finish_reason="stop", message=ChatCompletionMessage(
            role="assistant", content=content))],
    )


class RuntimeAccountingTests(unittest.TestCase):
    def setUp(self):
        self.model = AuditedOpenAIServerModel({
            "MODEL": "deepseek-v4.1-flash", "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
            "DEEPSEEK_API_KEY": "offline-test-only", "DEEPSEEK_MAX_RETRIES": "40",
            "DEEPSEEK_TIMEOUT": "300", "DEEPSEEK_DISABLE_THINKING": "all",
        }, max_api_attempts=2, request_timeout=1)
        self.messages = [{"role": "user", "content": [{"type": "text", "text": "offline"}]}]
        self.sleep = patch("hyperskill.no_skill_runtime.time.sleep", return_value=None)
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()
        self.model.client.close()

    def connection_error(self):
        # Use the installed SDK's own transport; SDK 3 uses httpx2, SDK 1/2 httpx.
        # build_request is local construction and never sends a request.
        request = self.model.client._client.build_request(
            "POST", "https://example.invalid/v1/chat/completions")
        return openai.APIConnectionError(request=request)

    def test_retry_exhaustion_is_bounded_and_raises(self):
        with patch.object(self.model.client.chat.completions, "create", side_effect=self.connection_error()) as create:
            with self.assertRaises(SafeRuntimeError) as failed:
                self.model(self.messages)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(safe_error(failed.exception), {
            "type": "SafeRuntimeError", "category": "llm_request_failed"})
        self.assertEqual(len(self.model.call_ledger), 2)
        self.assertTrue(all(row["usage"] is None for row in self.model.call_ledger))

    def test_empty_content_response_keeps_billed_usage_before_retry(self):
        with patch.object(self.model.client.chat.completions, "create", side_effect=[
            response("", 8, 2, 10), response("done", 12, 3, 15),
        ]) as create:
            answer = self.model(self.messages)
        self.assertEqual(answer.content, "done")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(self.model.call_ledger[0]["outcome"], "empty_content")
        summary = self.model.usage_summary()
        self.assertTrue(summary["usage_complete"])
        self.assertEqual(summary["total_usage"], {
            "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})

    def test_failed_attempt_usage_stays_unknown_even_after_success(self):
        with patch.object(self.model.client.chat.completions, "create", side_effect=[
            self.connection_error(), response("done"),
        ]):
            self.model(self.messages)
        summary = self.model.usage_summary()
        self.assertIsNone(self.model.call_ledger[0]["usage"])
        self.assertEqual(summary["unknown_usage_attempts"], [1])
        self.assertFalse(summary["usage_complete"])
        self.assertIsNone(summary["total_usage"])
        self.assertEqual(summary["known_usage"]["total_tokens"], 10)

    def test_inconsistent_negative_or_boolean_usage_cannot_be_complete(self):
        for usage in ((8, 2, 999), (-1, 2, 1), (True, 2, 3)):
            with self.subTest(usage=usage):
                with patch.object(self.model.client.chat.completions, "create", return_value=response("done", *usage)):
                    self.model(self.messages)
                summary = self.model.usage_summary()
                self.assertFalse(summary["usage_complete"])
                self.assertIsNone(summary["total_usage"])
                self.assertEqual(summary["known_usage"]["total_tokens"], 0)
                self.assertEqual(summary["invalid_usage_attempts"][-1], len(self.model.call_ledger))

    def test_no_request_is_not_complete_task_accounting(self):
        summary = self.model.usage_summary()
        self.assertFalse(summary["usage_complete"])
        self.assertIsNone(summary["total_usage"])


class NativeTerminationTests(unittest.TestCase):
    def test_null_final_call_and_max_step_fallback_are_not_success(self):
        called = ActionStep(step_number=1, tool_calls=[ToolCall(
            name="final_answer", arguments={"answer": None}, id="")])
        fallback = ActionStep(step_number=7, error=AgentMaxStepsError(
            "Reached max steps.", AgentLogger(LogLevel.OFF)))
        evidence = native_final_answer_evidence([called, fallback])
        self.assertTrue(evidence["called"])
        self.assertFalse(evidence["nonempty_answer"])
        self.assertTrue(evidence["max_steps_fallback"])
        self.assertEqual(evidence["step_errors"], [{"step_number": 7, "type": "AgentMaxStepsError"}])

    def test_actual_nonempty_final_answer_has_no_fallback(self):
        step = ActionStep(step_number=2, tool_calls=[ToolCall(
            name="final_answer", arguments={"answer": "verified"}, id="")])
        evidence = native_final_answer_evidence([step])
        self.assertTrue(evidence["nonempty_answer"])
        self.assertFalse(evidence["max_steps_fallback"])
        self.assertEqual(evidence["step_errors"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
