"""Connect native runner JSON evidence to the immutable SQLite store.

Historical import preserves the original observations and marks their provenance.
It neither reruns a task nor turns a source observation into a skill evaluation.
"""
import hashlib
import json
from pathlib import Path
import uuid

from .evidence_store import EvidenceStore, StoreValidation, StoreNotFound, SCHEMA_VERSION

BUNDLE_NAMES = (
    "result.json", "trajectory.json", "messages.json", "tool_events.json",
    "page_fetches.json", "api_calls.json",
)
OPTIONAL_NAMES = ("independent-validation.json",)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_bundle(directory):
    root = Path(directory)
    bundle = {}
    for name in BUNDLE_NAMES:
        path = root/name
        if not path.is_file() or path.is_symlink():
            raise StoreValidation("incomplete_run_bundle")
        bundle[name] = json.loads(path.read_text(encoding="utf-8"))
    for name in OPTIONAL_NAMES:
        if (root/name).is_file() and not (root/name).is_symlink():
            bundle[name] = json.loads((root/name).read_text(encoding="utf-8"))
    return bundle


def _validate_result(result):
    if not isinstance(result, dict) or result.get("mode") != "no_skill_task":
        raise StoreValidation("only_native_task_runs_are_imported")
    task = result.get("task")
    if not isinstance(task, str) or not task.strip():
        raise StoreValidation("task_text_missing")
    if result.get("task_sha256") != hashlib.sha256(task.encode()).hexdigest():
        raise StoreValidation("task_fingerprint_mismatch")
    config = result.get("configuration")
    if not isinstance(config, dict) or result.get("configuration_fingerprint") != digest(config):
        raise StoreValidation("protocol_fingerprint_mismatch")
    if (config.get("memory_provider") is not None
            or config.get("learned_skill_injection") is not False
            or config.get("learned_skill_retrieval") is not False):
        raise StoreValidation("bundle_is_not_no_skill_execution")
    return task, config


def new_execution_identity():
    run = uuid.uuid4().hex
    return {"task_instance_id": "task_"+run, "main_branch_id": "branch_"+run+"_main",
            "registration_mode": "live", "planning_recorded_before_execution": True}


def _historical_identity(bundle, source_claim=None):
    run = digest({name: bundle[name] for name in BUNDLE_NAMES})
    identity = {"task_instance_id": "task_import_"+run,
                "main_branch_id": "branch_import_"+run+"_main",
                "registration_mode": "historical_import",
                "planning_recorded_before_execution": False}
    if source_claim is not None:
        identity["source_identity_claim"] = source_claim
    return identity


def _identity(bundle):
    result = bundle["result.json"]
    explicit = result.get("execution_identity")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise StoreValidation("invalid_execution_identity")
        if explicit.get("registration_mode") != "live" or explicit.get("planning_recorded_before_execution") is not True:
            raise StoreValidation("invalid_execution_identity")
        for name in ("task_instance_id", "main_branch_id"):
            if not isinstance(explicit.get(name), str) or not explicit[name].strip():
                raise StoreValidation("invalid_execution_identity")
        return {name: explicit[name] for name in (
            "task_instance_id", "main_branch_id", "registration_mode", "planning_recorded_before_execution")}
    # Directory location and later independent checks do not change run identity.
    return _historical_identity(bundle)


def register_execution_task(store, result, identity):
    """Commit task identity before milestone generation, without opening a branch."""
    task, config = _validate_result(result)
    metadata = {"origin": "native_no_skill_runner",
                "registration_mode": identity["registration_mode"],
                "source_planning_recorded_before_execution": identity["planning_recorded_before_execution"]}
    if "source_identity_claim" in identity:
        metadata["source_identity_claim"] = identity["source_identity_claim"]
    with store.transaction():
        protocol_id = store.register_protocol(config, result["configuration_fingerprint"])
        store.register_task(identity["task_instance_id"], task,
                            main_branch_id=identity["main_branch_id"],
                            external_task_id=result.get("task_id"), metadata=metadata)
    return {"task_id": identity["task_instance_id"],
            "branch_id": identity["main_branch_id"], "protocol_id": protocol_id}


def register_execution_plan(store, result, identity):
    """Open the immutable branch with this task's already frozen specification.

    Imported pre-milestone runs retain their original null specification. Replaying
    a current run uses the same frozen reference instead of rewriting its plan.
    """
    with store.transaction():
        receipt = register_execution_task(store, result, identity)
        row = store.connection.execute(
            "SELECT spec_id FROM milestone_specs WHERE task_id=?",
            (receipt["task_id"],)).fetchone()
        spec_ref = row[0] if row else None
        if (identity['registration_mode'] == 'live' and 'milestone_spec_ref' in result
                and result['milestone_spec_ref'] != spec_ref):
            raise StoreValidation('execution_milestone_spec_mismatch')
        store.register_branch(receipt["branch_id"], receipt["task_id"],
                              receipt["protocol_id"], roles=["main"], supplied_versions=[],
                              milestone_spec_ref=spec_ref,
                              metadata={"purpose": "no_skill_execution"})
    return receipt


def _validate_usage_bundle(result, calls):
    summary = result.get("usage_summary")
    if summary is None:
        if calls:
            raise StoreValidation("api_calls_require_usage_summary")
        return
    if not isinstance(summary, dict):
        raise StoreValidation("invalid_usage_summary")
    count = summary.get("request_attempts")
    if type(count) is not int or count != len(calls):
        raise StoreValidation("api_attempt_count_mismatch")
    known = {k: 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    unknown, invalid = [], []
    for sequence, call in enumerate(calls, 1):
        if not isinstance(call, dict) or type(call.get("sequence")) is not int or call["sequence"] != sequence:
            raise StoreValidation("api_call_sequence_mismatch")
        usage = call.get("usage")
        if usage is None:
            unknown.append(sequence)
        elif (not isinstance(usage, dict)
                or any(type(usage.get(k)) is not int or usage[k] < 0 for k in known)
                or usage["prompt_tokens"]+usage["completion_tokens"] != usage["total_tokens"]):
            invalid.append(sequence)
        else:
            for key in known:
                known[key] += usage[key]
    complete = bool(calls) and not unknown and not invalid
    if (type(summary.get("usage_complete")) is not bool or summary["usage_complete"] != complete
            or summary.get("known_usage") != known
            or summary.get("unknown_usage_attempts") != unknown
            or summary.get("invalid_usage_attempts") != invalid):
        raise StoreValidation("api_usage_detail_mismatch")
    if complete and summary.get("total_usage") != known:
        raise StoreValidation("api_usage_total_mismatch")
    if not complete and summary.get("total_usage") is not None:
        raise StoreValidation("unknown_usage_must_remain_null")


def persist_bundle(store, bundle, historical=False):
    if any(name not in bundle for name in BUNDLE_NAMES):
        raise StoreValidation("incomplete_run_bundle")
    result = bundle["result.json"]
    _validate_result(result)
    for name in BUNDLE_NAMES[1:]:
        if not isinstance(bundle[name], list):
            raise StoreValidation("run_artifact_requires_list")
    _validate_usage_bundle(result, bundle["api_calls.json"])
    identity = _identity(bundle)
    importing = historical or identity["registration_mode"] == "historical_import"
    source_claim = identity if identity["registration_mode"] == "live" else None
    reused = False
    with store.transaction():
        if historical and source_claim is not None:
            # Importing a file cannot establish that its claimed live plan existed.
            # Reuse a matching existing plan, otherwise create a distinct import identity.
            existing = []
            for reader, key in ((store.get_task, "task_instance_id"),
                                (store.get_branch, "main_branch_id")):
                try:
                    reader(identity[key]); existing.append(True)
                except StoreNotFound:
                    existing.append(False)
            if any(existing) and not all(existing):
                raise StoreValidation("incomplete_existing_live_plan")
            if all(existing):
                reused = True
            else:
                identity = _historical_identity(bundle, source_claim)
        if identity["registration_mode"] == "live":
            # Native completion may only complete a plan already committed earlier.
            store.get_task(identity["task_instance_id"])
            store.get_branch(identity["main_branch_id"])
        receipt = register_execution_plan(store, result, identity)
        evidence = {}
        for name in BUNDLE_NAMES+OPTIONAL_NAMES:
            if name not in bundle:
                continue
            kind = "engineering_validation" if name in OPTIONAL_NAMES else name.removesuffix(".json")
            evidence_id = store.put_evidence(kind, bundle[name])
            store.attach_branch_evidence(receipt["branch_id"], name, evidence_id)
            evidence[name] = evidence_id
        store.record_branch_result(receipt["branch_id"], result,
                                   usage_summary=result.get("usage_summary"))
    return {"status": "stored", "schema_version": SCHEMA_VERSION, **receipt,
            "registration_mode": "historical_import" if importing else "live",
            "planning_recorded_before_execution": not importing,
            "existing_live_plan_reused": reused,
            "source_identity_claim": source_claim,
            "evidence_refs": evidence}


def import_run_directory(store, directory):
    return persist_bundle(store, load_bundle(directory), historical=True)
