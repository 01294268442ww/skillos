"""Evidence import and native runner integration; no external API requests."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from hyperskill.evidence_store import EvidenceStore, StoreConflict, StoreValidation, StoreError
from hyperskill.run_evidence import (
    BUNDLE_NAMES, digest, import_run_directory, new_execution_identity,
    persist_bundle, register_execution_plan,
)


def fixture():
    task = 'Engineering fixture, not an actual task execution.'
    config = {'model': {'model': 'fixture-model', 'endpoint': 'https://example.invalid/v1'},
              'memory_provider': None, 'learned_skill_injection': False,
              'learned_skill_retrieval': False, 'mode': 'no_skill_task'}
    usage = {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}
    summary = {'request_attempts': 1, 'usage_complete': True, 'known_usage': usage,
               'total_usage': usage, 'unknown_usage_attempts': [], 'invalid_usage_attempts': []}
    result = {'mode': 'no_skill_task', 'task_id': 'fixture-external-id', 'task': task,
              'task_sha256': hashlib.sha256(task.encode()).hexdigest(),
              'configuration': config, 'configuration_fingerprint': digest(config),
              'status': 'fixture_only', 'final_answer': 'fixture answer', 'usage_summary': summary}
    bundle = {name: [] for name in BUNDLE_NAMES}
    bundle['result.json'] = result
    bundle['api_calls.json'] = [{'sequence': 1, 'phase': 'agent', 'usage': usage}]
    return bundle


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'store/evidence.sqlite3'
        self.store = EvidenceStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_repeated_historical_import_is_one_run_and_preserves_unknowns(self):
        bundle = fixture()
        bundle['api_calls.json'] = [{'sequence': 1, 'usage': None}]
        bundle['result.json']['usage_summary'] = {
            'request_attempts': 1, 'usage_complete': False, 'total_usage': None,
            'known_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'unknown_usage_attempts': [1], 'invalid_usage_attempts': []}
        receipt = persist_bundle(self.store, bundle)
        counts = self.store.summary()
        self.assertEqual(persist_bundle(self.store, copy.deepcopy(bundle)), receipt)
        self.assertEqual(self.store.summary(), counts)
        branch = self.store.get_branch(receipt['branch_id'])
        self.assertIsNone(branch['usage_summary']['total_usage'])
        self.assertEqual(receipt['registration_mode'], 'historical_import')
        self.assertFalse(receipt['planning_recorded_before_execution'])

    def test_inconsistent_api_ledger_cannot_enter_store(self):
        bundle = fixture()
        bundle['api_calls.json'][0]['usage']['total_tokens'] = 999
        with self.assertRaises(StoreValidation):
            persist_bundle(self.store, bundle)
        self.assertTrue(all(n == 0 for n in self.store.summary()['counts'].values()))

    def test_conflicting_reimport_rolls_back_new_evidence(self):
        bundle = fixture()
        bundle['result.json']['execution_identity'] = new_execution_identity()
        register_execution_plan(self.store, bundle['result.json'], bundle['result.json']['execution_identity'])
        receipt = persist_bundle(self.store, bundle)
        before = self.store.summary()
        changed = copy.deepcopy(bundle)
        changed['result.json']['final_answer'] = 'conflicting observation'
        with self.assertRaises(StoreConflict):
            persist_bundle(self.store, changed)
        self.assertEqual(self.store.summary(), before)
        self.assertEqual(self.store.get_branch(receipt['branch_id'])['result']['final_answer'], 'fixture answer')

    def test_live_claim_requires_existing_plan_and_import_does_not_certify_it(self):
        bundle = fixture()
        bundle['result.json']['execution_identity'] = new_execution_identity()
        from hyperskill.evidence_store import StoreNotFound
        with self.assertRaises(StoreNotFound):
            persist_bundle(self.store, bundle)
        self.assertTrue(all(n == 0 for n in self.store.summary()['counts'].values()))
        receipt = persist_bundle(self.store, bundle, historical=True)
        self.assertEqual(receipt['registration_mode'], 'historical_import')
        self.assertFalse(receipt['planning_recorded_before_execution'])
        self.assertNotEqual(receipt['task_id'], bundle['result.json']['execution_identity']['task_instance_id'])
        stored = self.store.get_task(receipt['task_id'])
        self.assertEqual(stored['metadata']['source_identity_claim'], bundle['result.json']['execution_identity'])

    def test_import_of_existing_live_run_reuses_one_execution_without_rebilling(self):
        bundle = fixture()
        identity = new_execution_identity();bundle['result.json']['execution_identity'] = identity
        register_execution_plan(self.store, bundle['result.json'], identity)
        live = persist_bundle(self.store,bundle)
        before = self.store.summary()
        imported = persist_bundle(self.store,bundle,historical=True)
        self.assertEqual(imported['task_id'],live['task_id'])
        self.assertTrue(imported['existing_live_plan_reused'])
        self.assertFalse(imported['planning_recorded_before_execution'])
        self.assertEqual(self.store.summary(),before)

    def test_partial_usage_known_total_and_unknown_indices_match_raw_rows(self):
        for mutation in ('known','unknown'):
            with self.subTest(mutation=mutation):
                bundle = fixture()
                bundle['api_calls.json'].append({'sequence':2,'usage':None})
                summary = bundle['result.json']['usage_summary']
                summary.update(request_attempts=2,usage_complete=False,total_usage=None,unknown_usage_attempts=[2])
                if mutation == 'known':summary['known_usage']={'prompt_tokens':4,'completion_tokens':5,'total_tokens':9}
                else:summary['unknown_usage_attempts']=[]
                with self.assertRaises(StoreValidation):persist_bundle(self.store,bundle)
        self.assertTrue(all(n == 0 for n in self.store.summary()['counts'].values()))

    def test_independent_validation_can_be_added_without_duplicating_original_run(self):
        bundle = fixture()
        first = persist_bundle(self.store, bundle)
        bundle['independent-validation.json'] = {'kind': 'engineering_fixture', 'status': 'fixture_only'}
        second = persist_bundle(self.store, bundle)
        self.assertEqual(first['task_id'], second['task_id'])
        self.assertEqual(self.store.summary()['counts']['tasks'], 1)
        self.assertEqual(len(self.store.get_branch(first['branch_id'])['evidence_refs']), 7)

    def test_live_plan_survives_close_before_result_exists(self):
        result = fixture()['result.json']
        identity = new_execution_identity()
        receipt = register_execution_plan(self.store, result, identity)
        self.store.close()
        with EvidenceStore(self.path, create=False) as other:
            branch = other.get_branch(receipt['branch_id'])
            self.assertIsNone(branch['result'])
            self.assertIsNone(branch['usage_summary'])
            self.assertEqual(branch['roles'], ['main'])

    def test_cli_import_then_new_process_read_without_model(self):
        directory = Path(self.tmp.name)/'bundle';directory.mkdir()
        for name, payload in fixture().items():
            (directory/name).write_text(json.dumps(payload))
        cli = [sys.executable, str(ROOT/'runners/evidence_store.py'), '--db', str(self.path),
               '--config', str(Path(self.tmp.name)/'no-runtime-config')]
        imported = subprocess.run(cli+['import-run','--run-dir',str(directory)], check=True,
                                  capture_output=True, text=True)
        receipt = json.loads(imported.stdout)
        queried = subprocess.run(cli+['show-task','--task-id',receipt['task_id']], check=True,
                                 capture_output=True,text=True)
        document = json.loads(queried.stdout)
        self.assertEqual(document['branches'][0]['result'], fixture()['result.json'])
        self.assertEqual(len(document['evidence']), 6)


class NativeRunnerPersistenceTests(unittest.TestCase):
    def run_fixture(self, fail_commit=False):
        from base_agent import SearchAgent
        from hyperskill.no_skill_runtime import AuditedOpenAIServerModel
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); db = tmp/'state/evidence.sqlite3'
            task_file = tmp/'task.json'; task_file.write_text(json.dumps({'task':'Controlled offline failure fixture.'}))
            observed = []
            class FakeModel:
                def __init__(self, config, **kwargs):
                    self.public_configuration = {'model': 'offline-fixture', 'endpoint': 'https://example.invalid/v1'}
                    self.call_ledger = []
                    self.client = SimpleNamespace(close=lambda:None)
                def usage_summary(self):
                    return {'request_attempts':0,'usage_complete':False,'total_usage':None,
                            'known_usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0},
                            'unknown_usage_attempts':[],'invalid_usage_attempts':[]}
            class FakeAgent:
                def __init__(self, model, **kwargs):
                    with EvidenceStore(db,create=False) as check:
                        self_check = check.connection.execute('SELECT branch_id FROM branches').fetchone()[0]
                        planned = check.get_branch(self_check)
                        observed.append(planned['roles']==['main'] and planned['result'] is None)
                    self.agent_fn = SimpleNamespace(logger=SimpleNamespace(level=None), tools={},
                        memory=SimpleNamespace(steps=[]),memory_provider=None,
                        write_memory_to_messages=lambda **kwargs:[])
                def __call__(self, task, max_retries):
                    return {'error':'controlled_offline_failure'}
                def capture_trajectory(self):
                    return {'agent_trajectory':[]}
            spec = importlib.util.spec_from_file_location('native_runner_under_test', ROOT/'runners/run_no_skill.py')
            module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            argv = ['run_no_skill.py','--task-file',str(task_file),'--output-dir',str(tmp/'output'),
                    '--evidence-db',str(db)]
            config = {'MODEL':'offline-fixture','DEEPSEEK_BASE_URL':'https://example.invalid/v1',
                      'DEEPSEEK_API_KEY':'offline-fixture-key'}
            previous = logging.root.manager.disable
            try:
                # These pre-existing persistence fixtures isolate the new maintenance
                # boundary. Full milestone freezing/checking has separate real-SQL tests.
                def prepare(database, task_id, configuration, **kwargs):
                    from hyperskill.milestone_store import MilestoneStore
                    specification = {'milestones':[{'id':'m1','requirement':'Controlled offline fixture.',
                        'target_state':'An offline fixture answer is delivered.',
                        'evidence_rules':['Inspect the saved final answer.'],'dependencies':[],
                        'acceptance_time':'at_delivery','alternatives':[],'weight':'1',
                        'weight_reason':'Only fixture requirement.'}]}
                    with MilestoneStore(database,create=False) as state:
                        claim = state.claim_spec(task_id,{'model':'offline-fixture'})
                        receipt = state.commit_spec(task_id,claim['owner_token'],specification,[],json.dumps(specification))
                    receipt['maintenance_api'] = {'llm_calls':[],'usage':{}}
                    return receipt
                checked = {'status':'unknown','session_id':'offline-mock-check',
                           'maintenance_api':{'llm_calls':[],'usage':{}}}
                with patch('base_agent.SearchAgent',FakeAgent), patch('hyperskill.no_skill_runtime.AuditedOpenAIServerModel',FakeModel), patch('hyperskill.no_skill_runtime.load_runtime_config',return_value=config), patch('hyperskill.milestones.prepare_task_from_config',side_effect=prepare), patch('hyperskill.milestones.check_branches_from_config',return_value=copy.deepcopy(checked)), patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()) as printed:
                    if fail_commit:
                        with patch('hyperskill.run_evidence.persist_bundle', side_effect=StoreError('controlled_commit_failure')):
                            code = module.main()
                    else:
                        code = module.main()
                summary = json.loads(printed.getvalue().strip())
            finally:
                logging.disable(previous)
            with EvidenceStore(db,create=False) as check:
                branch_id = check.connection.execute('SELECT branch_id FROM branches').fetchone()[0]
                stored = check.get_branch(branch_id)
            receipt = json.loads((tmp/'output/storage_receipt.json').read_text())
            self.assertEqual(observed,[True])
            self.assertEqual(code,2)
            return summary,stored,receipt

    def test_failed_native_execution_is_stored_after_pre_registration(self):
        summary,stored,receipt = self.run_fixture()
        self.assertEqual(receipt['status'],'stored')
        self.assertEqual(stored['result']['status'],'failed_acceptance')
        self.assertIsNone(stored['usage_summary']['total_usage'])
        self.assertEqual(summary['status'],'failed_acceptance')

    def test_database_commit_failure_is_not_reported_as_success(self):
        summary,stored,receipt = self.run_fixture(fail_commit=True)
        self.assertEqual(receipt['status'],'error')
        self.assertEqual(summary['status'],'persistence_error')
        self.assertIsNone(stored['result'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
