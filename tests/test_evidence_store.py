"""Behavior checks for the persistent store; all data lives in temporary DBs.

Run with HYPERSKILL_REPO pointing at the checkout. No server or model requests.
"""
from contextlib import suppress
import importlib.util
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest


REPO = Path(os.environ.get("HYPERSKILL_REPO", Path(__file__).resolve().parents[1]))
MODULE_PATH = REPO / "hyperskill/evidence_store.py"
# Load this standard-library module directly, so tests do not depend on the
# package's unrelated eager imports or initialize an agent/memory provider.
spec = importlib.util.spec_from_file_location("tested_evidence_store", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
EvidenceStore = module.EvidenceStore
StoreError = module.StoreError
StoreConflict = module.StoreConflict
StoreValidation = module.StoreValidation
StoreNotFound = module.StoreNotFound


class EvidenceStoreBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hyperskill-store-test-")
        self.path = Path(self.temp.name) / "evidence.sqlite3"
        self.opened = []
        self.store = self.open_store()
        self.protocol = {"model": "deepseek-v4.1-flash", "memory_provider": None,
                         "max_tokens": 2048, "tools": ["crawl_page", "web_search"]}
        self.protocol_id = self.store.register_protocol(self.protocol)

    def tearDown(self):
        for store in reversed(self.opened):
            with suppress(Exception):
                store.close()
        self.temp.cleanup()

    def open_store(self, **kwargs):
        store = EvidenceStore(self.path, **kwargs)
        self.opened.append(store)
        return store

    def task_branch(self, task_id="q1", roles=None, task="Read the public source"):
        branch_id = task_id + ":main"
        self.store.register_task(task_id, task, main_branch_id=branch_id,
                                 external_task_id="shared-dataset-label", formal_round=None)
        self.store.register_branch(branch_id, task_id, self.protocol_id,
                                   roles=roles or ["main"], supplied_versions=[])
        return task_id, branch_id

    def version(self, skill_id="s1", version=1, content="Read and check the date",
                conditions=None, parents=()):
        self.store.put_skill(skill_id)
        return self.store.put_skill_version(
            skill_id, version, content,
            conditions={"page_kind": "official"} if conditions is None else conditions,
            goal_refs=["check-date"], origin_kind="manual_test_fixture",
            parent_versions=parents)

    def evidence(self, branch_id, payload=None, name="trajectory.json"):
        evidence_id = self.store.put_evidence(
            "execution", {"actions": [{"tool": "crawl_page"}]} if payload is None else payload)
        self.store.attach_branch_evidence(branch_id, name, evidence_id)
        return evidence_id

    def complete_usage(self, incoming=12, outgoing=3):
        totals = {"prompt_tokens": incoming, "completion_tokens": outgoing,
                  "total_tokens": incoming + outgoing}
        return {"request_attempts": 1, "usage_complete": True,
                "unknown_usage_attempts": [], "invalid_usage_attempts": [],
                "known_usage": totals, "total_usage": totals}

    def test_close_reopen_and_new_process_preserve_complete_snapshot(self):
        task_id, branch_id = self.task_branch()
        version_id = self.version()
        evidence_id = self.evidence(branch_id)
        self.store.link_version_evidence(version_id, "source", task_id, branch_id, evidence_id)
        self.store.record_branch_result(branch_id, {"status": "passed", "final_answer": "Verified"},
                                        usage_summary=self.complete_usage())
        expected = self.store.export_task(task_id)
        expected_version = self.store.get_skill_version(version_id)
        self.store.close()
        fresh = self.open_store(create=False)
        self.assertEqual(fresh.export_task(task_id), expected)
        self.assertEqual(fresh.get_skill_version(version_id), expected_version)
        code = """
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('child_store',sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
spec.loader.exec_module(module)
with module.EvidenceStore(sys.argv[2],create=False) as store:
 print(json.dumps({'task':store.export_task(sys.argv[3]),'version':store.get_skill_version(sys.argv[4])},sort_keys=True))
"""
        result = subprocess.run([sys.executable, "-c", code, str(MODULE_PATH), str(self.path),
                                 task_id, version_id], check=True, capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(json.loads(result.stdout), {"task": expected, "version": expected_version})

    def test_duplicate_full_writes_are_idempotent(self):
        task_id, branch_id = self.task_branch()
        version_id = self.version()
        evidence_id = self.evidence(branch_id)
        self.store.link_version_evidence(version_id, "source", task_id, branch_id, evidence_id)
        usage = self.complete_usage()
        self.store.record_branch_result(branch_id, {"status": "passed"}, usage_summary=usage)
        counts = self.store.summary()["counts"]
        self.assertEqual(self.store.register_protocol(self.protocol), self.protocol_id)
        self.task_branch()
        self.assertEqual(self.version(), version_id)
        self.assertEqual(self.evidence(branch_id), evidence_id)
        self.store.link_version_evidence(version_id, "source", task_id, branch_id, evidence_id)
        self.store.record_branch_result(branch_id, {"status": "passed"}, usage_summary=usage)
        self.assertEqual(self.store.summary()["counts"], counts)

    def test_conflicting_task_and_version_cannot_overwrite_existing_records(self):
        self.task_branch()
        version_id = self.version()
        task = self.store.get_task("q1")
        version = self.store.get_skill_version(version_id)
        with self.assertRaises(StoreConflict):
            self.store.register_task("q1", "A different task", main_branch_id="q1:main")
        with self.assertRaises(StoreConflict):
            self.version(content="Changed same numbered version")
        self.assertEqual(self.store.get_task("q1"), task)
        self.assertEqual(self.store.get_skill_version(version_id), version)

    def test_outer_transaction_rolls_back_all_partial_objects(self):
        counts = self.store.summary()["counts"]
        with self.assertRaises(StoreError):
            with self.store.transaction():
                self.store.register_task("atomic", "Must roll back", main_branch_id="atomic:main")
                self.store.put_skill("atomic-skill")
                self.store.register_branch("atomic:main", "atomic", "missing-protocol", roles=["main"])
        self.assertEqual(self.store.summary()["counts"], counts)
        with self.assertRaises(StoreNotFound):
            self.store.get_task("atomic")

    def test_new_content_and_conditions_have_new_fingerprints_and_null_values(self):
        parent = self.version()
        changed_content = self.version(version=2, content="Read and check date and year", parents=[parent])
        changed_conditions = self.version(version=3, conditions={"page_kind": "official", "year": 2026},
                                          parents=[changed_content])
        records = [self.store.get_skill_version(v) for v in (parent, changed_content, changed_conditions)]
        self.assertEqual(len({r["content_sha256"] for r in records}), 3)
        self.assertTrue(all(r["value"] is None for r in records))
        self.assertEqual(records[1]["parent_versions"], [parent])
        self.assertEqual(records[2]["parent_versions"], [changed_content])
        self.assertEqual(len(self.store.list_skill_versions("s1")), 3)
        derived = self.version(skill_id="derived", content="A separately identified derived skill",
                               parents=[parent, changed_conditions])
        self.assertEqual(set(self.store.get_skill_version(derived)["parent_versions"]),
                         {parent, changed_conditions})
        self.assertIsNone(self.store.get_skill_version(derived)["value"])

    def test_missing_exact_parent_is_rejected_without_partial_version(self):
        self.store.put_skill("child")
        counts = self.store.summary()["counts"]
        with self.assertRaises(StoreError):
            self.store.put_skill_version("child", 1, "new", {}, [], "manual_test_fixture",
                                          parent_versions=["does-not-exist"])
        self.assertEqual(self.store.summary()["counts"], counts)
        self.assertEqual(self.store.list_skill_versions("child"), [])

    def test_identical_prompt_does_not_merge_distinct_task_instances(self):
        self.task_branch("instance-one", task="Identical prompt")
        self.task_branch("instance-two", task="Identical prompt")
        self.assertNotEqual(self.store.get_task("instance-one")["task_id"],
                            self.store.get_task("instance-two")["task_id"])
        self.assertEqual(len(self.store.export_task("instance-one")["branches"]), 1)
        self.assertEqual(len(self.store.export_task("instance-two")["branches"]), 1)

    def test_main_and_base_roles_share_one_branch_and_one_cost_record(self):
        task_id, branch_id = self.task_branch(roles=["main", "base"])
        usage = self.complete_usage(5835, 712)
        self.store.record_branch_result(branch_id, {"status": "passed"}, usage_summary=usage)
        exported = self.store.export_task(task_id)
        self.assertEqual(len(exported["branches"]), 1)
        branch = exported["branches"][0]
        self.assertEqual(set(branch["roles"]), {"main", "base"})
        self.assertEqual(branch["usage_summary"]["total_usage"]["total_tokens"], 6547)
        self.assertEqual(self.store.summary()["counts"]["branch_results"], 1)

    def test_main_role_must_match_predeclared_main_branch(self):
        self.store.register_task("planned", "Task", main_branch_id="planned:main")
        with self.assertRaises(StoreError):
            self.store.register_task("other-task", "Other", main_branch_id="planned:main")
        with self.assertRaises(StoreError):
            self.store.register_branch("planned:extra", "planned", self.protocol_id, roles=["main"])
        with self.assertRaises(StoreError):
            self.store.register_branch("planned:main", "planned", self.protocol_id, roles=["base"])
        self.assertEqual(self.store.export_task("planned")["branches"], [])

    def test_sources_belong_to_exact_version_and_matching_task_branch(self):
        task_id, branch_id = self.task_branch("source-one")
        other_task, other_branch = self.task_branch("source-two")
        version1 = self.version()
        version2 = self.version(version=2, content="Changed", parents=[version1])
        evidence_id = self.evidence(branch_id)
        self.store.link_version_evidence(version1, "source", task_id, branch_id, evidence_id)
        self.assertEqual(len(self.store.get_skill_version(version1)["evidence_refs"]), 1)
        self.assertEqual(self.store.get_skill_version(version2)["evidence_refs"], [])
        before = self.store.summary()["counts"]
        with self.assertRaises(StoreError):
            self.store.link_version_evidence(version2, "source", other_task, branch_id, evidence_id)
        with self.assertRaises(StoreError):
            self.store.link_version_evidence(version2, "source", other_task, other_branch, evidence_id)
        self.assertEqual(self.store.summary()["counts"], before)

    def test_source_and_validation_roles_are_relative_to_version(self):
        task_id, branch_id = self.task_branch()
        version1 = self.version()
        version2 = self.version(skill_id="other-skill")
        evidence_id = self.evidence(branch_id)
        self.store.link_version_evidence(version1, "source", task_id, branch_id, evidence_id)
        self.store.link_version_evidence(version2, "validation", task_id, branch_id, evidence_id)
        first = self.store.get_skill_version(version1)["evidence_refs"]
        second = self.store.get_skill_version(version2)["evidence_refs"]
        self.assertEqual(first[0]["relation"], "source")
        self.assertEqual(second[0]["relation"], "validation")
        self.assertNotIn("task_type", self.store.get_task(task_id))

    def test_unknown_usage_remains_null_and_known_partial_usage_is_preserved(self):
        _, branch_id = self.task_branch()
        usage = {"request_attempts": 2, "usage_complete": False,
                 "unknown_usage_attempts": [1], "invalid_usage_attempts": [],
                 "known_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                 "total_usage": None}
        self.store.record_branch_result(branch_id, {"status": "error", "error": {"type": "NetworkError"}},
                                        usage_summary=usage)
        stored = self.store.get_branch(branch_id)
        self.assertEqual(stored["usage_summary"], usage)
        self.assertIsNone(stored["usage_summary"]["total_usage"])
        self.assertEqual(stored["result"]["status"], "error")

    def test_invalid_complete_usage_does_not_create_result(self):
        _, branch_id = self.task_branch()
        inconsistent_total = self.complete_usage()
        inconsistent_total["total_usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 999}
        inconsistent_known = self.complete_usage()
        inconsistent_known["known_usage"] = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        inconsistent_unknown = self.complete_usage()
        inconsistent_unknown["unknown_usage_attempts"] = [1]
        for usage in (inconsistent_total, inconsistent_known, inconsistent_unknown):
            with self.subTest(usage=usage):
                with self.assertRaises(StoreError):
                    self.store.record_branch_result(branch_id, {"status": "passed"}, usage_summary=usage)
        self.assertIsNone(self.store.get_branch(branch_id)["result"])

    def test_secret_values_and_sensitive_fields_are_rejected_without_leak(self):
        protected = "unit-private-key-value-never-store"
        guarded = self.open_store(secrets=[protected])
        attempts = [
            lambda: guarded.register_protocol({"model": protected}),
            lambda: guarded.register_task("unsafe", "task", main_branch_id="unsafe:main",
                                           metadata={"nested": [protected]}),
            lambda: guarded.put_evidence("execution", {"text": "prefix " + protected}),
            lambda: guarded.put_evidence("execution", {"api_key": "test-credential"}),
            lambda: guarded.register_protocol({"nested": {"Authorization": "Bearer test"}}),
        ]
        before = guarded.summary()["counts"]
        for write in attempts:
            with self.subTest(write=write):
                with self.assertRaises(StoreError) as caught:
                    write()
                self.assertNotIn(protected, str(caught.exception))
        self.assertEqual(guarded.summary()["counts"], before)
        for path in Path(self.temp.name).iterdir():
            if path.is_file():
                self.assertNotIn(protected.encode(), path.read_bytes())
        # Token accounting fields are public data, not credential fields.
        guarded.put_evidence("usage", {"prompt_tokens": 10, "completion_tokens": 2,
                                       "total_tokens": 12, "max_tokens": 2048})

    def test_shared_evidence_is_content_addressed_and_keeps_both_references(self):
        task1, branch1 = self.task_branch("one")
        task2, branch2 = self.task_branch("two")
        payload = {"fact": "same immutable public evidence"}
        evidence1 = self.evidence(branch1, payload)
        evidence2 = self.evidence(branch2, payload)
        self.assertEqual(evidence1, evidence2)
        changed = self.store.put_evidence("execution", {"fact": "different"})
        self.assertNotEqual(evidence1, changed)
        self.assertEqual(self.store.get_evidence(evidence1)["payload"], payload)
        self.assertEqual(self.store.get_branch(branch1)["evidence_refs"][0]["evidence_id"], evidence1)
        self.assertEqual(self.store.get_branch(branch2)["evidence_refs"][0]["evidence_id"], evidence1)
        self.assertEqual(self.store.summary()["counts"]["evidence"], 2)
        for task in (task1, task2):
            self.assertTrue(self.store.export_task(task)["evidence"])

    def concurrent_write(self, values):
        barrier = threading.Barrier(2)
        outcomes = queue.Queue()
        def worker(text):
            try:
                with EvidenceStore(self.path, create=False) as store:
                    barrier.wait(timeout=10)
                    store.register_task("racing", text, main_branch_id="racing:main")
                    outcomes.put(("ok", text))
            except Exception as error:
                outcomes.put((type(error).__name__, text))
        threads = [threading.Thread(target=worker, args=(text,), daemon=True) for text in values]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(any(thread.is_alive() for thread in threads), "Concurrent write deadlocked")
        return [outcomes.get_nowait() for _ in threads]

    def test_two_connections_same_payload_are_idempotent(self):
        outcomes = self.concurrent_write(["same", "same"])
        self.assertEqual([status for status, _ in outcomes], ["ok", "ok"])
        self.assertEqual(self.store.summary()["counts"]["tasks"], 1)

    def test_two_connections_conflicting_payload_have_one_winner(self):
        outcomes = self.concurrent_write(["first", "second"])
        self.assertEqual(sorted(status for status, _ in outcomes), ["StoreConflict", "ok"])
        winner = next(text for status, text in outcomes if status == "ok")
        self.assertEqual(self.store.get_task("racing")["task"], winner)
        self.assertEqual(self.store.summary()["counts"]["tasks"], 1)

    def test_unknown_schema_is_refused_without_destroying_existing_data(self):
        self.store.close()
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE future_marker (value TEXT)")
            db.execute("INSERT INTO future_marker VALUES ('keep')")
            db.execute("PRAGMA user_version=999")
        with self.assertRaises(StoreError):
            EvidenceStore(self.path, create=False)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 999)
            self.assertEqual(db.execute("SELECT value FROM future_marker").fetchone()[0], "keep")

    def test_invalid_json_and_boolean_version_are_rejected(self):
        for payload in ({"score": float("nan")}, {"score": float("inf")},
                        {"not_json": {1, 2}}, {"\ud800": "invalid UTF-8 key"}):
            with self.subTest(payload=payload):
                with self.assertRaises(StoreError):
                    self.store.put_evidence("execution", payload)
        self.store.put_skill("bad-version")
        with self.assertRaises(StoreError):
            self.store.put_skill_version("bad-version", True, "body", {}, [], "manual_test_fixture")

    def test_foreign_key_integrity_and_private_database_mode(self):
        task_id, branch_id = self.task_branch()
        version_id = self.version()
        evidence_id = self.evidence(branch_id)
        self.store.link_version_evidence(version_id, "source", task_id, branch_id, evidence_id)
        check = self.store.integrity_check()
        self.assertIn("ok", json.dumps(check).lower())
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
