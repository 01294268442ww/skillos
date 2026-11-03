#!/usr/bin/env python3
"""Run the official SearchAgent without learned memory and save real evidence."""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tool_succeeded(output):
    value = str(output).strip().lower()
    errors = ("error", "invalid url", "content extraction failed", "duckduckgo search failed",
              "no results found", "no search results found", "query is empty")
    return bool(value) and not value.startswith(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/home/ubuntu/.config/hyperskill/runtime.env")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-file")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence-db", default=str(ROOT/"storage/evidence.sqlite3"),
                        help="SQLite evidence store for task runs; probes/preflight are not formal tasks.")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-api-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--meta-max-tokens", type=int, default=4096,
                        help="Separate milestone/checker response budget; never charged to task C.")
    parser.add_argument("--meta-request-timeout", type=float, default=90)
    parser.add_argument("--preflight-url", default="https://peps.python.org/pep-0020/")
    parser.add_argument("--search-query", default="Python PEP 20 The Zen of Python")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "result.json").exists():
        print(json.dumps({"status": "error", "error": {"type": "OutputAlreadyExists"}}))
        return 2
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "probe" if args.probe else "preflight" if args.preflight else "no_skill_task",
              "protocol_status": "functional_smoke_not_final_experimental_B",
              "status": "error"}
    model = agent = None
    secrets = []
    tool_events = []
    page_fetches = []
    trajectory = []
    messages = []
    captured_logs = io.StringIO()
    store = None
    plan_registered = False
    persistence = None
    learning = None
    milestone = None
    checker = None
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(captured_logs), contextlib.redirect_stderr(captured_logs):
            from base_agent import SearchAgent
            from FlashOAgents.monitoring import LogLevel
            from FlashOAgents.search_tools import WebSearchTool, read_page, DIRECT_PAGE_FETCHES
            from hyperskill.no_skill_runtime import (
                AuditedOpenAIServerModel, SafeRuntimeError, apply_tool_environment,
                fingerprint, load_runtime_config, native_final_answer_evidence, redact, safe_error,
            )
            page_fetches = DIRECT_PAGE_FETCHES
            page_fetches.clear()
            config = load_runtime_config(args.config)
            secrets = [v for k, v in config.items() if v and any(
                marker in k.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))]
            apply_tool_environment(config)
            if args.max_steps < 1:
                raise SafeRuntimeError("max_steps_must_be_positive")
            model = AuditedOpenAIServerModel(
                config, max_tokens=args.max_tokens,
                max_api_attempts=args.max_api_attempts, request_timeout=args.request_timeout)
            protocol = {
                "model": model.public_configuration,
                "max_steps": args.max_steps, "summary_interval": 8,
                "disable_planning": True, "memory_provider": None,
                "learned_skill_retrieval": False, "learned_skill_injection": False,
                "web_access_provider": os.environ["WEB_ACCESS_PROVIDER"],
                "search_backend": (os.environ["WEB_SEARCH_PROVIDER"]
                                   if os.environ["WEB_SEARCH_PROVIDER"] != "auto"
                                   else "serper" if os.environ.get("SERPER_API_KEY") else "duckduckgo"),
                "prompt_sha256": hashlib.sha256((ROOT / "FlashOAgents/prompts/default/toolcalling_agent.yaml").read_bytes()).hexdigest(),
                "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (Path(__file__).resolve(), ROOT / "hyperskill/no_skill_runtime.py", ROOT / "FlashOAgents/search_tools.py",
                                                    ROOT / "hyperskill/evidence_store.py", ROOT / "hyperskill/run_evidence.py",
                                                    ROOT / "hyperskill/learning.py", ROOT / "hyperskill/learning_store.py",
                                                    ROOT / "hyperskill/api_embedding.py", ROOT / "hyperskill/milestones.py",
                                                    ROOT / "hyperskill/milestone_store.py")},
                "milestone_checker": {"max_tokens": args.meta_max_tokens,
                                      "request_timeout": args.meta_request_timeout,
                                      "max_api_attempts_per_round": 1, "max_check_rounds": 3,
                                      "single_milestone_scoring": "binary_with_unknown",
                                      "round_budget_scope": "frozen_branch_session",
                                      "cost_account": "maintenance"},
                "mode": result["mode"],
            }
            protocol = redact(protocol, secrets)
            result["configuration"] = protocol
            result["configuration_fingerprint"] = fingerprint(protocol)
            if args.probe:
                model.phase = "probe"
                message = model([{"role": "user", "content": [{"type": "text", "text": "Reply exactly HYPERSKILL_DEEPSEEK_OK"}]}], max_tokens=32)
                result["probe_response"] = message.content
                result["acceptance"] = {"exact_probe_response": message.content.strip() == "HYPERSKILL_DEEPSEEK_OK",
                                        "usage_recorded": bool(model.call_ledger and model.call_ledger[-1]["usage"])}
            elif args.preflight:
                search = WebSearchTool()(query=args.search_query)
                page = read_page(args.preflight_url)
                result["preflight"] = {"search_query": args.search_query, "search_output": search,
                                       "page_url": args.preflight_url, "page_output": page}
                result["acceptance"] = {"imports_passed": True,
                                        "search_succeeded": tool_succeeded(search),
                                        "page_succeeded": tool_succeeded(page)}
            else:
                task_path = Path(args.task_file)
                raw = task_path.read_text(encoding="utf-8")
                task_data = json.loads(raw) if task_path.suffix.lower() == ".json" else {"task": raw}
                task = task_data.get("task")
                if not isinstance(task, str) or not task.strip():
                    raise SafeRuntimeError("task_file_requires_nonempty_task_string")
                result.update({"task_id": task_data.get("task_id"), "task": task,
                               "task_sha256": hashlib.sha256(task.encode()).hexdigest()})
                from hyperskill.evidence_store import EvidenceStore
                from hyperskill.run_evidence import (
                    new_execution_identity, register_execution_task,
                    register_execution_plan, persist_bundle,
                )
                from hyperskill.milestones import prepare_task_from_config
                result["execution_identity"] = new_execution_identity()
                store = EvidenceStore(args.evidence_db, secrets=secrets)
                # Freeze standards before a branch or its execution evidence exists.
                task_plan = register_execution_task(store, result, result["execution_identity"])
                milestone = prepare_task_from_config(
                    args.evidence_db, task_plan['task_id'], config,
                    max_tokens=args.meta_max_tokens, max_api_attempts=1,
                    request_timeout=args.meta_request_timeout)
                if milestone.get('status') not in ('completed', 'already_completed'):
                    raise SafeRuntimeError('milestone_preparation_failed')
                result['milestone_spec_ref'] = milestone['spec_id']
                register_execution_plan(store, result, result["execution_identity"])
                plan_registered = True
                result['execution_started_at_utc'] = datetime.now(timezone.utc).isoformat()
                agent = SearchAgent(model, summary_interval=8, prompts_type="default",
                                    max_steps=args.max_steps, memory_provider=None,
                                    disable_planning=True)
                agent.agent_fn.logger.level = LogLevel.OFF
                for name, tool in agent.agent_fn.tools.items():
                    if name == "final_answer":
                        continue
                    original = tool.forward
                    def tracked(*call_args, _name=name, _original=original, **call_kwargs):
                        event = {"tool": _name, "arguments": call_kwargs or list(call_args),
                                 "api_sequence_before": len(model.call_ledger)}
                        previous_phase = model.phase
                        model.phase = "tool:" + _name
                        try:
                            value = _original(*call_args, **call_kwargs)
                            event.update({"status": "ok" if tool_succeeded(value) else "error", "output": value})
                            return value
                        except Exception as error:
                            event.update({"status": "error", "error": safe_error(error)})
                            raise SafeRuntimeError("tool_execution_failed") from None
                        finally:
                            model.phase = previous_phase
                            event["api_sequence_after"] = len(model.call_ledger)
                            tool_events.append(event)
                    tool.forward = tracked
                returned = agent(task, max_retries=1)
                trajectory = agent.capture_trajectory().get("agent_trajectory", [])
                messages = agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
                termination = native_final_answer_evidence(agent.agent_fn.memory.steps)
                result["termination"] = termination
                no_guidance = agent.agent_fn.memory_provider is None and all(
                    not getattr(s, "memory_guidance", None) for s in agent.agent_fn.memory.steps)
                result["final_answer"] = returned.get("agent_result")
                if returned.get("error"):
                    result["execution_error"] = {"type": "AgentReturnedError"}
                result["acceptance"] = {
                    "agent_returned_without_error": not bool(returned.get("error")),
                    "native_final_answer_called": termination["called"],
                    "native_final_answer_nonempty": termination["nonempty_answer"],
                    "no_max_steps_fallback": not termination["max_steps_fallback"],
                    "nonempty_final_answer": bool(returned.get("agent_result")),
                    "successful_crawl_tool": any(x["tool"] == "crawl_page" and x["status"] == "ok" for x in tool_events),
                    "successful_real_page_fetch": any(x.get("status") == "ok" for x in page_fetches),
                    "no_learned_skill_guidance": no_guidance,
                    "usage_recorded": bool(model.call_ledger and any(x["usage"] for x in model.call_ledger)),
                }
            if not args.preflight:
                result["acceptance"]["usage_complete"] = model.usage_summary()["usage_complete"]
            result["status"] = "passed" if all(result["acceptance"].values()) else "failed_acceptance"
    except Exception as error:
        result["error"] = safe_error(error) if "safe_error" in locals() else {"type": type(error).__name__}
    finally:
        if agent is not None:
            try:
                trajectory = agent.capture_trajectory().get("agent_trajectory", [])
                messages = agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
                result["termination"] = native_final_answer_evidence(agent.agent_fn.memory.steps)
            except Exception as error:
                result["trajectory_capture_error"] = safe_error(error)
        if model is not None:
            result["usage_summary"] = model.usage_summary()
        # Redact by replacement as a final defense even for page/model text.
        def clean(value):
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[REDACTED]")
                return value
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, dict):
                return {clean(str(k)): clean(v) for k, v in value.items()}
            return {"unserialized_type": type(value).__name__}
        artifacts = {"result.json": result, "trajectory.json": trajectory,
                     "messages.json": messages, "tool_events.json": tool_events,
                     "page_fetches.json": page_fetches,
                     "api_calls.json": model.call_ledger if model is not None else []}
        artifacts = clean(artifacts)
        for name, value in artifacts.items():
            with (output / name).open("x", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        if args.task_file:
            if milestone is not None:
                milestone_api = milestone.pop('maintenance_api', {'llm_calls': [], 'usage': {}})
                for name, payload in (("milestone_receipt.json", milestone),
                                      ("milestone_api_calls.json", milestone_api)):
                    with (output/name).open('x', encoding='utf-8') as handle:
                        json.dump(clean(payload), handle, ensure_ascii=False, indent=2)
                        handle.write('\n')
            try:
                if not plan_registered:
                    raise RuntimeError("plan_not_registered")
                persistence = persist_bundle(store, artifacts)
                persistence["database_path"] = str(Path(args.evidence_db).resolve())
            except Exception as error:
                persistence = {"status": "error", "error": {"type": type(error).__name__}}
                # Store categories are controlled identifiers, never arbitrary error text.
                if "EvidenceStore" in locals():
                    from hyperskill.evidence_store import StoreError
                    if isinstance(error, StoreError):
                        persistence["error"]["category"] = error.category
                if not plan_registered:
                    persistence["error"]["category"] = "plan_not_registered"
            finally:
                if store is not None:
                    store.close()
            with (output/"storage_receipt.json").open("x", encoding="utf-8") as handle:
                json.dump(clean(persistence), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if persistence.get('status') == 'stored':
                try:
                    from hyperskill.milestones import check_branches_from_config
                    checker = check_branches_from_config(
                        args.evidence_db, [persistence['branch_id']], config,
                        max_tokens=args.meta_max_tokens, max_api_attempts=1,
                        request_timeout=args.meta_request_timeout)
                except Exception as error:
                    from hyperskill.learning import safe_error as checker_error
                    checker = {'status': 'error', 'error': checker_error(error)}
                checker_api = checker.pop('maintenance_api', {'llm_calls': [], 'usage': {}})
                for name, payload in (("checker_receipt.json", checker),
                                      ("checker_api_calls.json", checker_api)):
                    with (output/name).open('x', encoding='utf-8') as handle:
                        json.dump(clean(payload), handle, ensure_ascii=False, indent=2)
                        handle.write('\n')
                try:
                    from hyperskill.learning import process_main_from_config
                    # Learning happens after immutable execution evidence is committed.
                    # Its calls/costs are maintenance evidence, not added to this task's C.
                    learning = process_main_from_config(args.evidence_db,persistence['branch_id'],config)
                except Exception as error:
                    from hyperskill.learning import safe_error as learning_error
                    learning = {'status':'error','error':learning_error(error)}
                maintenance = learning.pop('maintenance_api',{'llm_calls':[],'embedding_calls':[],'usage':{}})
                with (output/'maintenance_api_calls.json').open('x',encoding='utf-8') as handle:
                    json.dump(clean(maintenance),handle,ensure_ascii=False,indent=2);handle.write('\n')
                learning['maintenance_usage'] = maintenance['usage']
                with (output/'learning_receipt.json').open('x',encoding='utf-8') as handle:
                    json.dump(clean(learning),handle,ensure_ascii=False,indent=2);handle.write('\n')
        if model is not None:
            with contextlib.suppress(Exception):
                model.client.close()
    status = result["status"]
    if milestone is not None and milestone.get('status') not in ('completed', 'already_completed'):
        status = 'milestone_error'
    elif persistence is not None and persistence.get("status") != "stored":
        status = "persistence_error"
    elif checker is not None and checker.get('status') not in ('completed', 'unknown'):
        status = 'checker_error'
    elif learning is not None and learning.get('status') not in (
            'completed','already_completed','no_executable_trajectory','skipped_extra'):
        status = 'learning_error'
    print(json.dumps(clean({"status": status, "execution_status": result["status"],
                      "persistence": persistence, "output_directory": str(output),
                      "learning": learning,
                      "milestone": milestone, "checker": checker,
                      "acceptance": result.get("acceptance", {}),
                      "usage_summary": result.get("usage_summary", {}),
                      "error": result.get("error")}), ensure_ascii=False))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
