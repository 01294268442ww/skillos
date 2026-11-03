"""Immutable skill versions and execution evidence, using SQLite only.

This module stores observations and references. It does not extract, retrieve,
evaluate, select, replace, merge or delete skills. A new version has no value.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat

SCHEMA_VERSION = 3
APPLICATION_ID = 0x48534B56
V1_TABLES = ("protocols", "tasks", "branches", "branch_roles", "branch_supplied_versions", "skills",
          "skill_versions", "version_parents", "evidence", "branch_evidence",
          "branch_results", "version_evidence")
LEARNING_TABLES = ("embedding_profiles", "embedding_cache", "learning_jobs",
                  "learning_api_events", "graph_nodes", "graph_hyperedges", "graph_memberships")
MILESTONE_TABLES = ("milestone_jobs", "milestone_specs", "check_sessions",
                    "check_bindings", "check_rounds", "meta_api_events")
TABLES = V1_TABLES + LEARNING_TABLES + MILESTONE_TABLES
SENSITIVE_KEYS = {"api_key", "apikey", "password", "passwd", "secret", "token",
                  "authorization", "access_token", "refresh_token", "client_secret"}


class StoreError(RuntimeError):
    def __init__(self, category="store_error"):
        self.category = category if isinstance(category, str) and re.fullmatch(
            r"[a-z][a-z0-9_]{0,79}", category) else "store_error"
        super().__init__(self.category)


class StoreValidation(StoreError):
    pass


class StoreConflict(StoreError):
    pass


class StoreNotFound(StoreError):
    pass


def canonical_json(value):
    """Strict, deterministic JSON; no object coercion or nonfinite numbers."""
    def check(item):
        if type(item) is str:
            try:
                item.encode("utf-8")
            except UnicodeError:
                raise StoreValidation("invalid_json_string") from None
            return
        if item is None or type(item) in (bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for key, child in item.items():
                check(key)
                check(child)
            return
        raise StoreValidation("invalid_json_value")
    try:
        check(value)
    except RecursionError:
        raise StoreValidation("cyclic_or_overdeep_json") from None
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, UnicodeError):
        raise StoreValidation("invalid_json_value") from None


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS protocols(
 protocol_id TEXT PRIMARY KEY, data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(
 task_id TEXT PRIMARY KEY, main_branch_id TEXT NOT NULL UNIQUE, data_json TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branches(
 branch_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
 protocol_id TEXT NOT NULL REFERENCES protocols(protocol_id),
 target_version_id TEXT REFERENCES skill_versions(version_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branch_roles(
 branch_id TEXT NOT NULL REFERENCES branches(branch_id), role TEXT NOT NULL,
 PRIMARY KEY(branch_id,role));
CREATE TABLE IF NOT EXISTS branch_supplied_versions(
 branch_id TEXT NOT NULL REFERENCES branches(branch_id),
 version_id TEXT NOT NULL REFERENCES skill_versions(version_id),
 ordinal INTEGER NOT NULL CHECK(ordinal>=0),
 PRIMARY KEY(branch_id,version_id), UNIQUE(branch_id,ordinal));
CREATE TABLE IF NOT EXISTS skills(
 skill_id TEXT PRIMARY KEY, data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS skill_versions(
 version_id TEXT PRIMARY KEY, skill_id TEXT NOT NULL REFERENCES skills(skill_id),
 version INTEGER NOT NULL CHECK(version>0), content_sha256 TEXT NOT NULL,
 value_json TEXT DEFAULT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(skill_id,version));
CREATE TABLE IF NOT EXISTS version_parents(
 version_id TEXT NOT NULL REFERENCES skill_versions(version_id),
 parent_version_id TEXT NOT NULL REFERENCES skill_versions(version_id),
 PRIMARY KEY(version_id,parent_version_id), CHECK(version_id<>parent_version_id));
CREATE TABLE IF NOT EXISTS evidence(
 evidence_id TEXT PRIMARY KEY, data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branch_evidence(
 branch_id TEXT NOT NULL REFERENCES branches(branch_id), name TEXT NOT NULL,
 evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
 PRIMARY KEY(branch_id,name));
CREATE TABLE IF NOT EXISTS branch_results(
 branch_id TEXT PRIMARY KEY REFERENCES branches(branch_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS version_evidence(
 version_id TEXT NOT NULL REFERENCES skill_versions(version_id), relation TEXT NOT NULL,
 task_id TEXT NOT NULL REFERENCES tasks(task_id),
 branch_id TEXT NOT NULL REFERENCES branches(branch_id),
 evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
 PRIMARY KEY(version_id,relation,task_id,branch_id,evidence_id));
CREATE INDEX IF NOT EXISTS branches_task ON branches(task_id);
CREATE INDEX IF NOT EXISTS version_evidence_branch ON version_evidence(branch_id);
"""

LEARNING_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_profiles(
 profile_id TEXT PRIMARY KEY, data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embedding_cache(
 cache_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES embedding_profiles(profile_id),
 text_sha256 TEXT NOT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(profile_id,text_sha256));
CREATE TABLE IF NOT EXISTS learning_jobs(
 branch_id TEXT PRIMARY KEY REFERENCES branches(branch_id), job_id TEXT NOT NULL UNIQUE,
 owner_token TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('running','completed','failed')),
 checkpoint_evidence_id TEXT REFERENCES evidence(evidence_id),
 receipt_evidence_id TEXT REFERENCES evidence(evidence_id), error_evidence_id TEXT REFERENCES evidence(evidence_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS learning_api_events(
 event_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
 kind TEXT NOT NULL, evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_nodes(
 node_id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('subtask','skill')),
 version_id TEXT REFERENCES skill_versions(version_id), data_json TEXT NOT NULL, created_at TEXT NOT NULL,
 CHECK((kind='skill' AND version_id IS NOT NULL) OR (kind='subtask' AND version_id IS NULL)));
CREATE TABLE IF NOT EXISTS graph_hyperedges(
 hyperedge_id TEXT PRIMARY KEY, branch_id TEXT NOT NULL UNIQUE REFERENCES branches(branch_id),
 profile_id TEXT NOT NULL REFERENCES embedding_profiles(profile_id),
 cache_id TEXT NOT NULL REFERENCES embedding_cache(cache_id), data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_memberships(
 hyperedge_id TEXT NOT NULL REFERENCES graph_hyperedges(hyperedge_id),
 node_id TEXT NOT NULL REFERENCES graph_nodes(node_id), PRIMARY KEY(hyperedge_id,node_id));
CREATE INDEX IF NOT EXISTS graph_members_node ON graph_memberships(node_id);
CREATE INDEX IF NOT EXISTS graph_edges_profile ON graph_hyperedges(profile_id);
"""
MILESTONE_SCHEMA = """
CREATE TABLE IF NOT EXISTS milestone_jobs(
 task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), owner_token TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('running','completed','failed')),
 receipt_evidence_id TEXT REFERENCES evidence(evidence_id),
 error_evidence_id TEXT REFERENCES evidence(evidence_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS milestone_specs(
 spec_id TEXT PRIMARY KEY, task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
 evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS check_sessions(
 session_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
 spec_id TEXT NOT NULL REFERENCES milestone_specs(spec_id),
 state TEXT NOT NULL CHECK(state IN ('active','closed')),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS check_bindings(
 branch_id TEXT PRIMARY KEY REFERENCES branches(branch_id),
 session_id TEXT NOT NULL REFERENCES check_sessions(session_id));
CREATE TABLE IF NOT EXISTS check_rounds(
 session_id TEXT NOT NULL REFERENCES check_sessions(session_id),
 round_no INTEGER NOT NULL CHECK(round_no>=1 AND round_no<=3),
 owner_token TEXT NOT NULL UNIQUE, state TEXT NOT NULL CHECK(state IN ('running','completed','failed')),
 request_json TEXT NOT NULL, evidence_id TEXT REFERENCES evidence(evidence_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY(session_id,round_no));
CREATE TABLE IF NOT EXISTS meta_api_events(
 event_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('milestone_generation','milestone_check')),
 evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
 data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS check_sessions_task ON check_sessions(task_id);
CREATE INDEX IF NOT EXISTS meta_api_operation ON meta_api_events(operation_id);
"""
SCHEMA = V1_SCHEMA + LEARNING_SCHEMA + MILESTONE_SCHEMA


class EvidenceStore:
    def __init__(self, path, secrets=(), create=True):
        self.path = Path(path).expanduser()
        self.secrets = tuple(s for s in secrets if isinstance(s, str) and s)
        self._savepoint = 0
        self.connection = None
        try:
            existed = self.path.exists()
            if self.path.is_symlink():
                raise StoreValidation("store_path_must_not_be_symlink")
            if not existed:
                if not create:
                    raise StoreNotFound("store_not_found")
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    os.close(descriptor)
                except FileExistsError:
                    pass  # A concurrent initializer may have created the file.
            if not stat.S_ISREG(self.path.stat().st_mode):
                raise StoreValidation("store_path_must_be_regular_file")
            self.connection = sqlite3.connect(self.path.resolve().as_uri()+"?mode=rw",
                                              uri=True, timeout=5, isolation_level=None)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA busy_timeout=5000")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            application = self.connection.execute("PRAGMA application_id").fetchone()[0]
            tables = {r[0] for r in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if version == 0 and not tables and create:
                self.connection.executescript("BEGIN IMMEDIATE;\n"+SCHEMA+
                    "\nPRAGMA application_id="+str(APPLICATION_ID)+";\nPRAGMA user_version=3;\nCOMMIT;")
            elif version == 1 and application == APPLICATION_ID and tables == set(V1_TABLES):
                # Additive migrations never rewrite an existing observation.
                self.connection.executescript("BEGIN IMMEDIATE;\n"+LEARNING_SCHEMA+MILESTONE_SCHEMA+
                    "\nPRAGMA user_version=3;\nCOMMIT;")
            elif version == 2 and application == APPLICATION_ID and tables == set(V1_TABLES+LEARNING_TABLES):
                self.connection.executescript("BEGIN IMMEDIATE;\n"+MILESTONE_SCHEMA+
                    "\nPRAGMA user_version=3;\nCOMMIT;")
            elif version != SCHEMA_VERSION or application != APPLICATION_ID or tables != set(TABLES):
                raise StoreValidation("unsupported_store_schema")
            self.path.chmod(0o600)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        except (OSError, sqlite3.Error):
            self.close()
            raise StoreError("store_open_failed") from None
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @contextmanager
    def transaction(self):
        if self.connection is None:
            raise StoreError("store_closed")
        nested = self.connection.in_transaction
        self._savepoint += 1
        name = "store_"+str(self._savepoint)
        try:
            self.connection.execute("SAVEPOINT "+name if nested else "BEGIN IMMEDIATE")
            yield self
            self.connection.execute("RELEASE "+name if nested else "COMMIT")
        except Exception as error:
            if self.connection.in_transaction:
                if nested:
                    self.connection.execute("ROLLBACK TO "+name)
                    self.connection.execute("RELEASE "+name)
                else:
                    self.connection.execute("ROLLBACK")
            if isinstance(error, sqlite3.IntegrityError):
                raise StoreValidation("foreign_key_or_constraint_violation") from None
            if isinstance(error, sqlite3.Error):
                raise StoreError("sqlite_operation_failed") from None
            raise

    def _validate(self, value):
        canonical_json(value)
        def inspect(item):
            if isinstance(item, str):
                if any(secret in item for secret in self.secrets) or re.search(
                        r"\bsk[-_][A-Za-z0-9_-]{12,}|\bBearer\s+[A-Za-z0-9._~-]{12,}", item, re.I):
                    raise StoreValidation("secret_value_not_allowed")
            elif isinstance(item, list):
                for child in item:
                    inspect(child)
            elif isinstance(item, dict):
                for key, child in item.items():
                    lower = key.lower().replace("-", "_")
                    sensitive = lower in SENSITIVE_KEYS or lower.endswith(("_api_key", "_password", "_secret", "_access_token", "_refresh_token"))
                    empty = child is None or child is False or (isinstance(child, str) and child in ("", "[REDACTED]", "<redacted>"))
                    if sensitive and not empty:
                        raise StoreValidation("secret_field_not_allowed")
                    inspect(key)
                    inspect(child)
        inspect(value)
        return canonical_json(value)

    def _id(self, value):
        if not isinstance(value, str) or not value.strip():
            raise StoreValidation("nonempty_identifier_required")
        self._validate(value)
        return value

    def _require(self, table, key, value):
        row = self.connection.execute("SELECT * FROM "+table+" WHERE "+key+"=?", (value,)).fetchone()
        if row is None:
            raise StoreNotFound("referenced_record_not_found")
        return row

    def _put(self, table, key, value, data, extra=None):
        serialized = self._validate(data)
        previous = self.connection.execute("SELECT data_json FROM "+table+" WHERE "+key+"=?", (value,)).fetchone()
        if previous:
            if previous[0] != serialized:
                raise StoreConflict("immutable_record_conflict")
            return False
        columns = {key: value, "data_json": serialized, "created_at": _now(), **(extra or {})}
        self.connection.execute("INSERT INTO "+table+"("+",".join(columns)+") VALUES("+
                                ",".join("?" for _ in columns)+")", tuple(columns.values()))
        return True

    def register_protocol(self, configuration, claimed_fingerprint=None):
        if not isinstance(configuration, dict):
            raise StoreValidation("configuration_must_be_object")
        self._validate(configuration)
        identifier = fingerprint(configuration)
        if claimed_fingerprint is not None and claimed_fingerprint != identifier:
            raise StoreValidation("protocol_fingerprint_mismatch")
        with self.transaction():
            self._put("protocols", "protocol_id", identifier, configuration)
        return identifier

    def register_task(self, task_id, task, main_branch_id, external_task_id=None,
                      formal_round=None, metadata=None):
        self._id(task_id); self._id(main_branch_id)
        if not isinstance(task, str) or not task.strip() or (formal_round is not None and
                (type(formal_round) is not int or formal_round < 1)):
            raise StoreValidation("invalid_task")
        if external_task_id is not None:
            self._id(external_task_id)
        if metadata is not None and not isinstance(metadata, dict):
            raise StoreValidation("metadata_must_be_object")
        self._validate(task)
        data = {"task_id": task_id, "task": task, "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
                "main_branch_id": main_branch_id, "external_task_id": external_task_id,
                "formal_round": formal_round, "metadata": metadata if metadata is not None else {}}
        with self.transaction():
            branch = self.connection.execute("SELECT task_id FROM branches WHERE branch_id=?", (main_branch_id,)).fetchone()
            if branch is not None and branch[0] != task_id:
                raise StoreValidation("main_branch_plan_mismatch")
            self._put("tasks", "task_id", task_id, data, {"main_branch_id": main_branch_id})
        return task_id

    def register_branch(self, branch_id, task_id, protocol_id, roles, operation=None,
                        target_version_id=None, supplied_versions=(), initial_state_ref=None,
                        milestone_spec_ref=None, metadata=None):
        for value in (branch_id, task_id, protocol_id):
            self._id(value)
        if not isinstance(roles, (list, tuple)) or not roles or any(not isinstance(x, str) or not x.strip() for x in roles):
            raise StoreValidation("branch_roles_required")
        roles = sorted(set(roles))
        if (not isinstance(supplied_versions, (list, tuple)) or
                any(not isinstance(x, str) or not x.strip() for x in supplied_versions) or
                len(set(supplied_versions)) != len(supplied_versions)):
            raise StoreValidation("invalid_supplied_versions")
        if operation is not None:
            self._id(operation)
        if metadata is not None and not isinstance(metadata, dict):
            raise StoreValidation("metadata_must_be_object")
        supplied_versions = list(supplied_versions)
        data = {"branch_id": branch_id, "task_id": task_id, "protocol_id": protocol_id,
                "roles": roles, "operation": operation, "target_version_id": target_version_id,
                "supplied_versions": supplied_versions, "initial_state_ref": initial_state_ref,
                "milestone_spec_ref": milestone_spec_ref, "metadata": metadata if metadata is not None else {}}
        with self.transaction():
            task = self._require("tasks", "task_id", task_id)
            frozen = self.connection.execute(
                "SELECT spec_id,evidence_id FROM milestone_specs WHERE task_id=?", (task_id,)).fetchone()
            if frozen is not None and milestone_spec_ref != frozen['spec_id']:
                raise StoreValidation("branch_must_use_frozen_milestones")
            if frozen is None and milestone_spec_ref is not None:
                raise StoreValidation("branch_milestone_spec_not_frozen")
            planned = self.connection.execute("SELECT task_id FROM tasks WHERE main_branch_id=?", (branch_id,)).fetchone()
            if planned is not None and planned[0] != task_id:
                raise StoreValidation("main_branch_plan_mismatch")
            if (branch_id == task["main_branch_id"]) != ("main" in roles):
                raise StoreValidation("main_branch_plan_mismatch")
            self._require("protocols", "protocol_id", protocol_id)
            for version in supplied_versions + ([target_version_id] if target_version_id is not None else []):
                self._id(version); self._require("skill_versions", "version_id", version)
            if self._put("branches", "branch_id", branch_id, data,
                         {"task_id": task_id, "protocol_id": protocol_id, "target_version_id": target_version_id}):
                self.connection.executemany("INSERT INTO branch_roles VALUES(?,?)", [(branch_id, r) for r in roles])
                self.connection.executemany("INSERT INTO branch_supplied_versions VALUES(?,?,?)",
                    [(branch_id, identifier, ordinal) for ordinal, identifier in enumerate(supplied_versions)])
            if frozen is not None:
                self.attach_branch_evidence(branch_id, 'milestone_spec', frozen['evidence_id'])
        return branch_id

    def put_evidence(self, kind, payload):
        self._id(kind)
        data = {"kind": kind, "payload": payload}
        self._validate(data)
        identifier = fingerprint(data)
        with self.transaction():
            self._put("evidence", "evidence_id", identifier, data)
        return identifier

    def attach_branch_evidence(self, branch_id, name, evidence_id):
        for value in (branch_id, name, evidence_id):
            self._id(value)
        with self.transaction():
            previous = self.connection.execute("SELECT evidence_id FROM branch_evidence WHERE branch_id=? AND name=?", (branch_id, name)).fetchone()
            if previous and previous[0] != evidence_id:
                raise StoreConflict("branch_evidence_conflict")
            if not previous:
                self.connection.execute("INSERT INTO branch_evidence VALUES(?,?,?)", (branch_id, name, evidence_id))
        return evidence_id

    def record_branch_result(self, branch_id, result, usage_summary=None):
        self._id(branch_id)
        if not isinstance(result, dict):
            raise StoreValidation("result_must_be_object")
        if usage_summary is None:
            usage_summary = result.get("usage_summary")
        if usage_summary is not None:
            if not isinstance(usage_summary, dict) or type(usage_summary.get("usage_complete")) is not bool:
                raise StoreValidation("invalid_usage_summary")
            def triplet(data):
                return (isinstance(data, dict) and all(type(data.get(k)) is int and data[k] >= 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")) and
                        data["prompt_tokens"] + data["completion_tokens"] == data["total_tokens"])
            if usage_summary["usage_complete"]:
                if not triplet(usage_summary.get("total_usage")):
                    raise StoreValidation("invalid_complete_usage")
                if any(usage_summary.get(key) for key in ("unknown_usage_attempts", "invalid_usage_attempts")):
                    raise StoreValidation("inconsistent_complete_usage")
                if ("request_attempts" in usage_summary and
                        (type(usage_summary["request_attempts"]) is not int or usage_summary["request_attempts"] < 1)):
                    raise StoreValidation("inconsistent_complete_usage")
                if "known_usage" in usage_summary and usage_summary["known_usage"] != usage_summary["total_usage"]:
                    raise StoreValidation("inconsistent_complete_usage")
            elif usage_summary.get("total_usage") is not None:
                raise StoreValidation("unknown_usage_requires_null_total")
            if "known_usage" in usage_summary and not triplet(usage_summary["known_usage"]):
                raise StoreValidation("invalid_known_usage")
            if "usage_summary" in result and result["usage_summary"] != usage_summary:
                raise StoreValidation("inconsistent_usage_summary")
        with self.transaction():
            self._put("branch_results", "branch_id", branch_id,
                      {"result": result, "usage_summary": usage_summary})
        return branch_id

    def put_skill(self, skill_id):
        self._id(skill_id)
        with self.transaction():
            self._put("skills", "skill_id", skill_id, {"skill_id": skill_id})
        return skill_id

    def put_skill_version(self, skill_id, version, content, conditions, goal_refs,
                          origin_kind, parent_versions=()):
        self._id(skill_id); self._id(origin_kind)
        if type(version) is not int or version < 1 or not isinstance(content, str) or not content.strip() or not isinstance(goal_refs, list):
            raise StoreValidation("invalid_skill_version")
        if not isinstance(parent_versions, (list, tuple)) or any(not isinstance(x, str) for x in parent_versions):
            raise StoreValidation("invalid_parent_versions")
        parents = sorted(set(parent_versions))
        identifier = "sv_"+fingerprint({"skill_id": skill_id, "version": version})
        digest = fingerprint({"content": content, "conditions": conditions, "goal_refs": goal_refs})
        data = {"version_id": identifier, "skill_id": skill_id, "version": version,
                "content": content, "conditions": conditions, "goal_refs": goal_refs,
                "origin_kind": origin_kind, "parent_versions": parents, "content_sha256": digest}
        with self.transaction():
            self._require("skills", "skill_id", skill_id)
            for parent in parents:
                self._id(parent)
                if parent == identifier:
                    raise StoreValidation("version_cannot_parent_itself")
                self._require("skill_versions", "version_id", parent)
            if self._put("skill_versions", "version_id", identifier, data,
                         {"skill_id": skill_id, "version": version, "content_sha256": digest}):
                self.connection.executemany("INSERT INTO version_parents VALUES(?,?)", [(identifier, p) for p in parents])
        return identifier

    def link_version_evidence(self, version_id, relation, task_id, branch_id, evidence_id):
        if relation not in ("source", "development", "validation", "diagnosis"):
            raise StoreValidation("unsupported_evidence_relation")
        for value in (version_id, task_id, branch_id, evidence_id):
            self._id(value)
        with self.transaction():
            self._require("skill_versions", "version_id", version_id)
            branch = self._require("branches", "branch_id", branch_id)
            if branch["task_id"] != task_id:
                raise StoreValidation("evidence_task_branch_mismatch")
            if not self.connection.execute("SELECT 1 FROM branch_evidence WHERE branch_id=? AND evidence_id=?", (branch_id, evidence_id)).fetchone():
                raise StoreValidation("evidence_not_attached_to_branch")
            values = (version_id, relation, task_id, branch_id, evidence_id)
            if not self.connection.execute("SELECT 1 FROM version_evidence WHERE version_id=? AND relation=? AND task_id=? AND branch_id=? AND evidence_id=?", values).fetchone():
                self.connection.execute("INSERT INTO version_evidence VALUES(?,?,?,?,?)", values)
        return evidence_id

    def _read(self, table, key, identifier):
        self._id(identifier)
        row = self._require(table, key, identifier)
        data = json.loads(row["data_json"])
        data["created_at"] = row["created_at"]
        return data

    def get_protocol(self, protocol_id):
        self._id(protocol_id)
        row = self._require("protocols", "protocol_id", protocol_id)
        return {"protocol_id": protocol_id, "configuration": json.loads(row["data_json"]), "created_at": row["created_at"]}

    def get_task(self, task_id):
        data = self._read("tasks", "task_id", task_id)
        data["branch_ids"] = [r[0] for r in self.connection.execute("SELECT branch_id FROM branches WHERE task_id=? ORDER BY branch_id", (task_id,))]
        return data

    def get_evidence(self, evidence_id):
        data = self._read("evidence", "evidence_id", evidence_id)
        return {"evidence_id": evidence_id, **data}

    def get_branch(self, branch_id):
        data = self._read("branches", "branch_id", branch_id)
        result = self.connection.execute("SELECT data_json,created_at FROM branch_results WHERE branch_id=?", (branch_id,)).fetchone()
        data.update(json.loads(result[0]) if result else {"result": None, "usage_summary": None})
        data["evidence_refs"] = [{"name": r[0], "evidence_id": r[1], "kind": json.loads(r[2])["kind"]}
            for r in self.connection.execute("SELECT b.name,b.evidence_id,e.data_json FROM branch_evidence b JOIN evidence e USING(evidence_id) WHERE b.branch_id=? ORDER BY b.name", (branch_id,))]
        return data

    def get_skill_version(self, version_id):
        data = self._read("skill_versions", "version_id", version_id)
        value = self._require("skill_versions", "version_id", version_id)["value_json"]
        data["value"] = json.loads(value) if value is not None else None
        data["evidence_refs"] = [dict(r) for r in self.connection.execute("SELECT relation,task_id,branch_id,evidence_id FROM version_evidence WHERE version_id=? ORDER BY relation,task_id,branch_id,evidence_id", (version_id,))]
        return data

    def list_skill_versions(self, skill_id):
        self._id(skill_id); self._require("skills", "skill_id", skill_id)
        return [self.get_skill_version(r[0]) for r in self.connection.execute("SELECT version_id FROM skill_versions WHERE skill_id=? ORDER BY version", (skill_id,))]

    def export_task(self, task_id):
        task = self.get_task(task_id)
        branches = [self.get_branch(identifier) for identifier in task["branch_ids"]]
        protocols = {b["protocol_id"]: self.get_protocol(b["protocol_id"]) for b in branches}
        evidence = {r["evidence_id"]: self.get_evidence(r["evidence_id"]) for b in branches for r in b["evidence_refs"]}
        return {"schema_version": SCHEMA_VERSION, "task": task, "branches": branches,
                "protocols": protocols, "evidence": evidence}

    def summary(self):
        return {"schema_version": SCHEMA_VERSION, "counts": {table: self.connection.execute(
            "SELECT COUNT(*) FROM "+table).fetchone()[0] for table in TABLES}}

    def integrity_check(self):
        return {"integrity": [r[0] for r in self.connection.execute("PRAGMA integrity_check")],
                "foreign_key_violations": [list(r) for r in self.connection.execute("PRAGMA foreign_key_check")]}
