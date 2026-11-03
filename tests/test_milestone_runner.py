"""Offline native-runner wiring with real SQLite and explicit fake API ledgers."""
import contextlib
import copy
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from hyperskill.evidence_store import EvidenceStore
from hyperskill.milestone_store import MilestoneStore


def ledger(tokens, phase):
    return [{'sequence': 1, 'phase': phase, 'outcome': 'response_received',
             'usage': {'prompt_tokens': tokens-1, 'completion_tokens': 1, 'total_tokens': tokens}}]


def summary(tokens):
    counts = {'prompt_tokens': tokens-1, 'completion_tokens': 1, 'total_tokens': tokens}
    return {'request_attempts': 1, 'usage_complete': True, 'total_usage': counts,
            'known_usage': counts, 'unknown_usage_attempts': [], 'invalid_usage_attempts': []}


def spec():
    return {'milestones': [{'id': 'm1', 'requirement': 'Deliver the offline fixture answer.',
        'target_state': 'The fixture answer is present in the final delivery.',
        'evidence_rules': ['Inspect the captured final answer.'], 'dependencies': [],
        'acceptance_time': 'at_delivery', 'alternatives': [], 'weight': '1',
        'weight_reason': 'This is the only fixture requirement.'}]}


class NativeMilestoneWiringTests(unittest.TestCase):
    def run_fixture(self, preparation_failure=False, checker_status='completed'):
        from FlashOAgents.search_tools import DIRECT_PAGE_FETCHES
        events = []
        with tempfile.TemporaryDirectory(prefix='milestone-native-fixture-') as temporary:
            root = Path(temporary)
            db = root/'state/evidence.sqlite3'
            task = root/'task.json'
            task.write_text(json.dumps({'task': 'Deliver the offline fixture answer.'}))
            config = {'MODEL': 'offline-model-fixture', 'DEEPSEEK_BASE_URL': 'https://example.invalid/v1',
                      'DEEPSEEK_API_KEY': 'offline-secret-fixture-value'}

            class FakeModel:
                def __init__(self, configuration, **kwargs):
                    self.call_ledger = []
                    self.public_configuration = {'model': 'offline-model-fixture',
                                                 'endpoint': 'https://example.invalid/v1'}
                    self.phase = 'offline_fixture'
                    self.client = SimpleNamespace(close=lambda: None)

                def usage_summary(self):
                    if self.call_ledger:
                        return summary(3)
                    return {'request_attempts': 0, 'usage_complete': False, 'total_usage': None,
                            'known_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                            'unknown_usage_attempts': [], 'invalid_usage_attempts': []}

            class FakeAgent:
                def __init__(self, model, **kwargs):
                    with MilestoneStore(db, create=False) as store:
                        task_id, branch_id = store.connection.execute(
                            'SELECT task_id,main_branch_id FROM tasks').fetchone()
                        frozen = store.get_spec(task_id)
                        branch = store.get_branch(branch_id)
                        assert branch['milestone_spec_ref'] == frozen['spec_id']
                        assert branch['result'] is None
                    assert model.call_ledger == []
                    events.append('agent_constructed_after_freeze')
                    self.model = model
                    self.final_call = SimpleNamespace(name='final_answer', arguments={'answer': 'offline fixture answer'})
                    self.steps = [SimpleNamespace(step_number=1, error=None, tool_calls=[self.final_call])]
                    def crawl_page(**kwargs):
                        DIRECT_PAGE_FETCHES.append({'status': 'ok', 'url': 'https://example.invalid/fixture',
                                                   'body': 'OFFLINE FIXTURE ONLY; no HTTP request.'})
                        return 'OFFLINE FIXTURE ONLY; no HTTP request.'
                    self.agent_fn = SimpleNamespace(
                        logger=SimpleNamespace(level=None),
                        tools={'crawl_page': SimpleNamespace(forward=crawl_page)},
                        memory=SimpleNamespace(steps=self.steps), memory_provider=None,
                        write_memory_to_messages=lambda **kwargs: [])

                def __call__(self, text, max_retries):
                    events.append('agent_executed')
                    self.model.call_ledger.extend(ledger(3, 'agent'))
                    self.agent_fn.tools['crawl_page'].forward(url='https://example.invalid/fixture')
                    return {'agent_result': 'offline fixture answer'}

                def capture_trajectory(self):
                    return {'agent_trajectory': [{'name': 'action', 'tool_calls': [],
                        'obs': 'OFFLINE FIXTURE ONLY; no HTTP request.'}]}

            def prepare(database, task_id, runtime, **kwargs):
                self.assertEqual(kwargs['max_api_attempts'], 1)
                with MilestoneStore(database, create=False) as store:
                    self.assertEqual(store.get_task(task_id)['branch_ids'], [])
                    events.append('prepare_before_branch')
                    if preparation_failure:
                        return {'status': 'error', 'error': {'category': 'offline_preparation_failure'},
                                'maintenance_api': {'llm_calls': ledger(5, 'milestone_generation'),
                                                    'usage': summary(5)}}
                    claim = store.claim_spec(task_id, {'model': 'offline-fixture-only'})
                    output = store.commit_spec(task_id, claim['owner_token'], spec(),
                                               ledger(5, 'milestone_generation'), json.dumps(spec()))
                    output['maintenance_api'] = {'llm_calls': ledger(5, 'milestone_generation'),
                                                 'usage': summary(5)}
                    return output

            def check(database, branches, runtime, **kwargs):
                self.assertEqual(kwargs['max_api_attempts'], 1)
                with MilestoneStore(database, create=False) as store:
                    self.assertEqual(store.get_branch(branches[0])['result']['final_answer'],
                                     'offline fixture answer')
                events.append('check_after_execution_persisted')
                return {'status': checker_status, 'session_id': 'offline-check-fixture',
                        'maintenance_api': {'llm_calls': ledger(7, 'milestone_checker'),
                                            'usage': summary(7)}}

            def learn(database, branch, runtime):
                events.append('learn_once_after_check')
                return {'status': 'completed', 'maintenance_api': {'llm_calls': ledger(11, 'routine_skill_extraction'),
                    'embedding_calls': [], 'usage': {'llm': summary(11), 'embedding': None}}}

            module_spec = importlib.util.spec_from_file_location('milestone_native_fixture', ROOT/'runners/run_no_skill.py')
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            argv = ['run_no_skill.py', '--task-file', str(task), '--evidence-db', str(db),
                    '--output-dir', str(root/'output')]
            old_logging = logging.root.manager.disable
            try:
                with patch('base_agent.SearchAgent', FakeAgent), patch('hyperskill.no_skill_runtime.AuditedOpenAIServerModel', FakeModel), patch('hyperskill.no_skill_runtime.load_runtime_config', return_value=config), patch('hyperskill.milestones.prepare_task_from_config', side_effect=prepare), patch('hyperskill.milestones.check_branches_from_config', side_effect=check), patch('hyperskill.learning.process_main_from_config', side_effect=learn), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as printed:
                    code = module.main()
                final = json.loads(printed.getvalue())
            finally:
                logging.disable(old_logging)
            artifacts = {path.name: json.loads(path.read_text()) for path in (root/'output').glob('*.json')}
            with EvidenceStore(db, create=False) as store:
                counts = store.summary()['counts']
                branches = [store.get_branch(row[0]) for row in store.connection.execute('SELECT branch_id FROM branches')]
            return code, final, artifacts, events, counts, branches

    def test_freeze_then_execute_persist_check_and_extract_with_separate_costs(self):
        code, final, artifacts, events, counts, branches = self.run_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(events, ['prepare_before_branch', 'agent_constructed_after_freeze',
            'agent_executed', 'check_after_execution_persisted', 'learn_once_after_check'])
        self.assertEqual(artifacts['result.json']['usage_summary']['total_usage']['total_tokens'], 3)
        self.assertEqual(len(artifacts['api_calls.json']), 1)
        self.assertEqual(artifacts['milestone_api_calls.json']['usage']['total_usage']['total_tokens'], 5)
        self.assertEqual(artifacts['checker_api_calls.json']['usage']['total_usage']['total_tokens'], 7)
        self.assertEqual(artifacts['maintenance_api_calls.json']['usage']['llm']['total_usage']['total_tokens'], 11)
        self.assertEqual(counts['milestone_specs'], 1)
        self.assertEqual(branches[0]['milestone_spec_ref'], artifacts['result.json']['milestone_spec_ref'])
        self.assertNotIn('phi', branches[0]['result'])

    def test_preparation_failure_keeps_paid_ledger_and_never_starts_branch(self):
        code, final, artifacts, events, counts, branches = self.run_fixture(preparation_failure=True)
        self.assertEqual(code, 2)
        self.assertEqual(final['status'], 'milestone_error')
        self.assertEqual(events, ['prepare_before_branch'])
        self.assertEqual(counts['branches'], 0)
        self.assertEqual(artifacts['api_calls.json'], [])
        self.assertEqual(artifacts['milestone_api_calls.json']['usage']['total_usage']['total_tokens'], 5)
        self.assertNotIn('checker_receipt.json', artifacts)
        self.assertNotIn('learning_receipt.json', artifacts)

    def test_closed_unknown_checker_still_extracts_main_exactly_once(self):
        code, final, artifacts, events, counts, branches = self.run_fixture(checker_status='unknown')
        self.assertEqual(code, 0)
        self.assertEqual(final['checker']['status'], 'unknown')
        self.assertEqual(events.count('learn_once_after_check'), 1)
        self.assertEqual(artifacts['result.json']['usage_summary']['total_usage']['total_tokens'], 3)

    def test_checker_error_keeps_execution_and_still_extracts_main(self):
        code, final, artifacts, events, counts, branches = self.run_fixture(checker_status='error')
        self.assertEqual(code, 2)
        self.assertEqual(final['status'], 'checker_error')
        self.assertEqual(final['execution_status'], 'passed')
        self.assertEqual(branches[0]['result']['status'], 'passed')
        self.assertEqual(events.count('learn_once_after_check'), 1)


class MilestoneCliReadTests(unittest.TestCase):
    def load_cli(self):
        definition = importlib.util.spec_from_file_location('milestone_cli_offline_fixture',
                                                          ROOT/'runners/milestone_check.py')
        module = importlib.util.module_from_spec(definition)
        definition.loader.exec_module(module)
        return module

    def test_show_frozen_specification_needs_no_runtime_config_or_model(self):
        with tempfile.TemporaryDirectory(prefix='milestone-cli-fixture-') as temporary:
            temporary = Path(temporary)
            db = temporary/'state/evidence.sqlite3'
            with MilestoneStore(db) as store:
                store.register_task('offline-task', 'Deliver an offline fixture.', 'offline-main')
                claim = store.claim_spec('offline-task', {'model': 'offline-only'})
                store.commit_spec('offline-task', claim['owner_token'], spec(), [], json.dumps(spec()))
            cli = self.load_cli()
            argv = ['milestone_check.py', '--db', str(db), '--config', str(temporary/'missing-runtime.env'),
                    '--output-dir', str(temporary/'output'), 'show-spec', '--task-id', 'offline-task']
            old_logging = logging.root.manager.disable
            try:
                with patch('hyperskill.no_skill_runtime.AuditedOpenAIServerModel') as constructor, patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(), 0)
                constructor.assert_not_called()
            finally:
                logging.disable(old_logging)
            operation = json.loads((temporary/'output/operation.json').read_text())
            self.assertEqual(operation['specification']['spec'], spec())
            self.assertEqual(json.loads((temporary/'output/llm_calls.json').read_text()), [])

    def test_existing_output_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory(prefix='milestone-cli-existing-') as temporary:
            directory = Path(temporary)/'existing'
            directory.mkdir()
            marker = directory/'preserved.txt'
            marker.write_text('Preserve the existing deliverable.')
            cli = self.load_cli()
            argv = ['milestone_check.py', '--output-dir', str(directory),
                    'show-spec', '--task-id', 'offline-task']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 2)
            self.assertEqual(marker.read_text(), 'Preserve the existing deliverable.')
            self.assertEqual([path.name for path in directory.iterdir()], ['preserved.txt'])


class ExecutionSpecificationBindingTests(unittest.TestCase):
    def test_live_plan_replays_the_frozen_spec_and_rejects_a_different_claim(self):
        import hashlib
        from hyperskill.evidence_store import StoreValidation
        from hyperskill.run_evidence import (
            digest, new_execution_identity, register_execution_task, register_execution_plan)
        with tempfile.TemporaryDirectory(prefix='milestone-plan-binding-') as temporary:
            with MilestoneStore(Path(temporary)/'state/evidence.sqlite3') as store:
                task = 'Deliver an offline fixture answer.'
                configuration = {'memory_provider': None, 'learned_skill_retrieval': False,
                                 'learned_skill_injection': False, 'model': 'offline-fixture-only'}
                result = {'mode': 'no_skill_task', 'task': task,
                    'task_sha256': hashlib.sha256(task.encode()).hexdigest(),
                    'configuration': configuration, 'configuration_fingerprint': digest(configuration)}
                identity = new_execution_identity()
                registered = register_execution_task(store, result, identity)
                claim = store.claim_spec(registered['task_id'], {'model': 'offline-fixture-only'})
                frozen = store.commit_spec(registered['task_id'], claim['owner_token'], spec(), [], json.dumps(spec()))
                result['milestone_spec_ref'] = 'another-specification-claim'
                with self.assertRaises(StoreValidation):
                    register_execution_plan(store, result, identity)
                self.assertEqual(store.summary()['counts']['branches'], 0)
                result['milestone_spec_ref'] = frozen['spec_id']
                register_execution_plan(store, result, identity)
                counts = store.summary()
                register_execution_plan(store, copy.deepcopy(result), copy.deepcopy(identity))
                self.assertEqual(store.summary(), counts)
                self.assertEqual(store.get_branch(registered['branch_id'])['milestone_spec_ref'], frozen['spec_id'])

    def test_imported_spec_claim_does_not_certify_a_new_databases_historical_run(self):
        import hashlib
        from hyperskill.evidence_store import StoreNotFound
        from hyperskill.run_evidence import (BUNDLE_NAMES, digest, new_execution_identity,
            register_execution_task, register_execution_plan, persist_bundle)
        with tempfile.TemporaryDirectory(prefix='milestone-historical-binding-') as temporary:
            temporary = Path(temporary)
            task = 'Deliver the offline fixture answer.'
            configuration = {'memory_provider': None, 'learned_skill_retrieval': False,
                             'learned_skill_injection': False, 'model': 'offline-fixture-only'}
            identity = new_execution_identity()
            result = {'mode': 'no_skill_task', 'task': task,
                      'task_sha256': hashlib.sha256(task.encode()).hexdigest(),
                      'configuration': configuration, 'configuration_fingerprint': digest(configuration),
                      'execution_identity': identity}
            with MilestoneStore(temporary/'live/evidence.sqlite3') as live:
                registered = register_execution_task(live, result, identity)
                claim = live.claim_spec(registered['task_id'], {'model': 'offline-fixture-only'})
                frozen = live.commit_spec(registered['task_id'], claim['owner_token'], spec(), [], json.dumps(spec()))
                result['milestone_spec_ref'] = frozen['spec_id']
                register_execution_plan(live, result, identity)
                bundle = {name: [] for name in BUNDLE_NAMES}
                bundle['result.json'] = result
                original = persist_bundle(live, bundle)
                before = live.summary()
                reimported = persist_bundle(live, copy.deepcopy(bundle), historical=True)
                self.assertEqual(reimported['branch_id'], original['branch_id'])
                self.assertTrue(reimported['existing_live_plan_reused'])
                self.assertEqual(live.summary(), before)
            with MilestoneStore(temporary/'historical/evidence.sqlite3') as historical:
                imported = persist_bundle(historical, copy.deepcopy(bundle), historical=True)
                branch = historical.get_branch(imported['branch_id'])
                self.assertFalse(imported['planning_recorded_before_execution'])
                self.assertIsNone(branch['milestone_spec_ref'])
                self.assertEqual(branch['result']['milestone_spec_ref'], frozen['spec_id'])
                with self.assertRaises(StoreNotFound):
                    historical.get_spec(imported['task_id'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
