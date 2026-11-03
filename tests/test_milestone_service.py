"""Milestone-service behavior using real temporary SQLite and offline API fixtures.

No SDK client or network call is made. The fixtures test the shared checker budget,
evidence ownership, immutable outcomes, failure accounting and task-only planning.
"""
from contextlib import suppress
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

REPO = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
PACKAGE = 'tested_milestone_service_package'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(REPO / 'hyperskill')]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(
        PACKAGE + '.' + name, REPO / 'hyperskill' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load('evidence_store')
storage = load('milestone_store')
service = load('milestones')


def specification():
    def milestone(identifier, weight):
        return {'id': identifier, 'requirement': 'Deliver fixture ' + identifier,
                'target_state': 'Fixture output states ' + identifier,
                'evidence_rules': ['Read the captured final answer for this output.'],
                'dependencies': [], 'acceptance_time': 'at_delivery', 'alternatives': [],
                'weight': weight, 'weight_reason': 'Importance of this requested output.'}
    return {'milestones': [milestone('m1', '0.6'), milestone('m2', '0.4')]}


class FixtureApiError(RuntimeError):
    pass


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls, self.call_ledger = [], []
        self.phase = 'offline_fixture'
        self.public_configuration = {
            'model': 'offline-deepseek-fixture', 'endpoint': 'https://example.invalid/v1',
            'effective': {'max_total_attempts': 1}}
        self.client = types.SimpleNamespace(close=Mock())

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError('Unexpected extra model request outside fixture budget')
        value = self.responses.pop(0)
        if callable(value):
            value = value(messages)
        row = {'sequence': len(self.call_ledger) + 1, 'phase': self.phase,
               'usage': None if isinstance(value, Exception) else {
                   'prompt_tokens': 9, 'completion_tokens': 2, 'total_tokens': 11},
               'outcome': 'request_failed' if isinstance(value, Exception) else 'response_received'}
        self.call_ledger.append(row)
        if isinstance(value, Exception):
            raise value
        return types.SimpleNamespace(content=value)

    def usage_summary(self):
        known = [row for row in self.call_ledger if row['usage'] is not None]
        complete = bool(known) and len(known) == len(self.call_ledger)
        totals = {key: sum(row['usage'][key] for row in known)
                  for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        return {'request_attempts': len(self.call_ledger), 'usage_complete': complete,
                'known_usage': totals, 'total_usage': totals if complete else None,
                'unknown_usage_attempts': [row['sequence'] for row in self.call_ledger
                                           if row['usage'] is None]}


class MilestoneServiceBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='hyperskill-milestone-service-test-')
        self.path = Path(self.tmp.name) / 'evidence.sqlite3'
        self.store = storage.MilestoneStore(self.path)
        self.protocol = self.store.register_protocol({'model': 'offline-fixture'})
        self.task_id, self.main_id = 'fixture-task', 'fixture-task:main'
        self.store.register_task(self.task_id, 'Deliver fixture m1 and fixture m2.',
                                 main_branch_id=self.main_id)
        self.evidence = {}

    def tearDown(self):
        with suppress(Exception):
            self.store.close()
        self.tmp.cleanup()

    def frozen(self):
        planner = service.MilestoneService(self.store, FakeModel([json.dumps(specification())]))
        return planner.prepare(self.task_id)

    def branch(self, branch_id=None, spec=None, historical=False):
        branch_id = branch_id or self.main_id
        self.store.register_branch(branch_id, self.task_id, self.protocol,
            roles=['main'] if branch_id == self.main_id else ['base'],
            milestone_spec_ref=None if historical else spec['spec_id'])
        payload = {'final_answer': 'Offline fixture output for ' + branch_id,
                   'status': 'engineering_fixture_only_not_a_checker_verdict'}
        ref = self.store.put_evidence('result', payload)
        self.store.attach_branch_evidence(branch_id, 'result.json', ref)
        self.store.record_branch_result(branch_id, payload)
        self.evidence[branch_id] = ref
        return branch_id

    @staticmethod
    def checker_source(messages):
        return json.loads(messages[0]['content'][0]['text'].split(
            '\nFrozen acceptance and evidence:\n', 1)[1])

    def judgments(self, messages, statuses=None):
        source = self.checker_source(messages)
        rows = []
        for request in source['requests']:
            branch = request['branch_id']
            for milestone in request['milestone_ids']:
                status = (statuses or {}).get((branch, milestone), 'unknown')
                rows.append({'branch_id': branch, 'milestone_id': milestone,
                    'status': status, 'reason': 'Controlled offline evidence interpretation.',
                    'evidence_refs': [] if status == 'unknown' else [{
                        'evidence_id': self.evidence[branch], 'pointer': '/final_answer'}],
                    'unknown_reason': 'The fixture lacks decisive evidence.' if status == 'unknown' else None})
        return {'judgments': rows}

    def response(self, statuses=None):
        return lambda messages: json.dumps(self.judgments(messages, statuses))

    def test_prepare_reads_only_task_requirements_and_freezes_before_any_branch(self):
        model = FakeModel([json.dumps(specification())])
        prepared = service.MilestoneService(self.store, model).prepare(self.task_id)
        self.assertEqual(prepared['status'], 'completed')
        prompt = model.calls[0][0]['content'][0]['text']
        source = json.loads(prompt.split('\nTask requirements:\n', 1)[1])
        self.assertEqual(source, {'task_id': self.task_id,
                                  'task': 'Deliver fixture m1 and fixture m2.'})
        self.assertEqual(self.store.get_task(self.task_id)['branch_ids'], [])
        self.assertTrue(prepared['frozen_before_any_branch'])
        self.assertEqual(model.call_ledger[0]['phase'], 'milestone_construction')
        self.assertNotIn('final_answer', source)
        self.assertNotIn('supplied_versions', source)

    def test_prepare_repeat_is_zero_calls_and_does_not_rewrite_frozen_spec(self):
        model = FakeModel([json.dumps(specification())])
        meta = service.MilestoneService(self.store, model)
        first = meta.prepare(self.task_id)
        self.branch(spec=first)
        before = self.store.summary()
        repeated = meta.prepare(self.task_id)
        self.assertEqual(repeated['status'], 'already_completed')
        self.assertEqual(repeated['new_llm_calls'], 0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(repeated['spec_id'], first['spec_id'])
        self.assertEqual(self.store.summary(), before)

    def test_historical_branch_rejects_prepare_before_model_request(self):
        self.branch(historical=True)
        model = FakeModel([])
        with self.assertRaises(core.StoreValidation) as caught:
            service.MilestoneService(self.store, model).prepare(self.task_id)
        self.assertEqual(caught.exception.category, 'milestones_must_precede_all_branches')
        self.assertEqual(model.calls, [])
        self.assertEqual(self.store.summary()['counts']['milestone_specs'], 0)

    def test_two_branches_share_three_rounds_and_unknown_is_not_zero(self):
        spec = self.frozen()
        one = self.branch(spec=spec)
        two = self.branch('fixture-task:extra', spec=spec)
        statuses = {(one, 'm1'): 'met', (one, 'm2'): 'unmet', (two, 'm1'): 'unmet'}
        model = FakeModel([self.response(statuses), self.response(), self.response()])
        meta = service.MilestoneService(self.store, model)
        result = meta.check([two, one])
        self.assertEqual((result['status'], result['round_count'], len(model.calls)), ('unknown', 3, 3))
        self.assertTrue(result['closed'])
        self.assertEqual(result['branches'][one]['phi'], '0.6')
        self.assertIsNone(result['branches'][two]['phi'])
        self.assertEqual(result['branches'][two]['known_subtotal'], '0')
        self.assertEqual(result['branches'][two]['unknown_ids'], ['m2'])
        for messages in model.calls[1:]:
            source = self.checker_source(messages)
            self.assertEqual(source['requests'], [{'branch_id': two, 'milestone_ids': ['m2']}])
            self.assertEqual(set(source['branches']), {two})
            self.assertEqual(source['spec_id'], spec['spec_id'])
        before = self.store.summary()
        repeated = meta.check([one, two])
        self.assertEqual(repeated['new_llm_calls'], 0)
        self.assertEqual(repeated['round_count'], 3)
        self.assertEqual(self.store.summary(), before)

    def test_unmet_cannot_be_flipped_by_overbroad_recheck_response(self):
        spec = self.frozen()
        branch = self.branch(spec=spec)
        def attempts_flip(messages):
            value = self.judgments(messages)
            value['judgments'].append({'branch_id': branch, 'milestone_id': 'm1',
                'status': 'met', 'reason': 'Invalid attempt to rewrite an earlier outcome.',
                'evidence_refs': [{'evidence_id': self.evidence[branch], 'pointer': '/final_answer'}],
                'unknown_reason': None})
            return json.dumps(value)
        model = FakeModel([self.response({(branch, 'm1'): 'unmet'}), attempts_flip,
                           self.response({(branch, 'm2'): 'met'})])
        result = service.MilestoneService(self.store, model).check([branch])
        judgments = {item['milestone_id']: item for item in result['branches'][branch]['judgments']}
        self.assertEqual(judgments['m1']['status'], 'unmet')
        self.assertEqual(judgments['m2']['status'], 'met')
        self.assertEqual(result['branches'][branch]['phi'], '0.4')
        self.assertEqual(result['rounds'][1]['state'], 'failed')
        self.assertEqual(result['rounds'][1]['error']['category'], 'checker_judgment_scope_mismatch')

    def test_invalid_json_and_api_error_consume_rounds_and_preserve_accounting(self):
        spec = self.frozen()
        branch = self.branch(spec=spec)
        model = FakeModel(['not valid json', FixtureApiError('offline request failure'), self.response()])
        result = service.MilestoneService(self.store, model).check([branch])
        self.assertEqual(result['round_count'], 3)
        self.assertEqual([item['state'] for item in result['rounds']], ['failed', 'failed', 'completed'])
        self.assertIsNone(result['branches'][branch]['phi'])
        ledgers = [self.store.get_evidence(ref)['payload']['calls'] for ref in result['llm_ledger_refs']]
        calls = [row for rows in ledgers for row in rows]
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(row['usage']['total_tokens'] for row in calls if row['usage']), 22)
        self.assertEqual(sum(row['usage'] is None for row in calls), 1)
        self.assertEqual(model.usage_summary()['total_usage'], None)
        self.assertEqual(result['rounds'][0]['raw_response'], 'not valid json')

    def test_cross_branch_reference_and_bad_pointer_cannot_establish_completion(self):
        spec = self.frozen()
        branch = self.branch(spec=spec)
        other = self.branch('fixture-task:extra', spec=spec)
        def wrong_branch(messages):
            value = self.judgments(messages, {(branch, 'm1'): 'met', (branch, 'm2'): 'met'})
            for item in value['judgments']:
                item['evidence_refs'][0]['evidence_id'] = self.evidence[other]
            return json.dumps(value)
        def wrong_pointer(messages):
            value = self.judgments(messages, {(branch, 'm1'): 'met', (branch, 'm2'): 'met'})
            for item in value['judgments']:
                item['evidence_refs'][0]['pointer'] = '/unrecorded/file'
            return json.dumps(value)
        model = FakeModel([wrong_branch, wrong_pointer, self.response()])
        result = service.MilestoneService(self.store, model).check([branch])
        self.assertEqual([item['state'] for item in result['rounds']], ['failed', 'failed', 'completed'])
        self.assertEqual([item['error']['category'] for item in result['rounds'][:2]],
                         ['checker_evidence_wrong_branch', 'evidence_pointer_not_found'])
        self.assertIsNone(result['branches'][branch]['phi'])
        self.assertTrue(all(item['status'] == 'unknown' for item in result['branches'][branch]['judgments']))

    def test_all_known_outcomes_finish_in_one_round_and_repeat_uses_no_api(self):
        spec = self.frozen()
        branch = self.branch(spec=spec)
        model = FakeModel([self.response({(branch, 'm1'): 'met', (branch, 'm2'): 'unmet'})])
        meta = service.MilestoneService(self.store, model)
        result = meta.check([branch])
        self.assertEqual((result['status'], result['round_count']), ('completed', 1))
        self.assertEqual(result['branches'][branch]['phi'], '0.6')
        before = self.store.summary()
        repeated = meta.check([branch])
        self.assertEqual(repeated['new_llm_calls'], 0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.store.summary(), before)

    def test_failed_spec_keeps_paid_usage_and_requires_explicit_retry(self):
        model = FakeModel(['{"milestones":[]} ', json.dumps(specification())])
        meta = service.MilestoneService(self.store, model)
        with self.assertRaises(core.StoreValidation):
            meta.prepare(self.task_id)
        job = self.store.get_spec_job(self.task_id)
        self.assertEqual(job['state'], 'failed')
        error = self.store.get_evidence(job['error_evidence_id'])['payload']
        ledger = self.store.get_evidence(error['llm_ledger_refs'][0])['payload']['calls']
        self.assertEqual(ledger[0]['usage']['total_tokens'], 11)
        self.assertEqual(meta.prepare(self.task_id)['status'], 'failed')
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(meta.prepare(self.task_id, retry_failed=True)['status'], 'completed')
        self.assertEqual(len(model.calls), 2)

    def test_db_failure_in_failure_handler_still_exports_paid_ledger_from_helper(self):
        spec = self.frozen()
        branch = self.branch(spec=spec)
        model = FakeModel(['bad JSON'])
        runtime = types.ModuleType(PACKAGE + '.no_skill_runtime')
        constructor = Mock(return_value=model)
        runtime.AuditedOpenAIServerModel = constructor
        runtime.redact = lambda value, secrets: value
        with patch.dict(sys.modules, {runtime.__name__: runtime}):
            with patch.object(storage.MilestoneStore, 'fail_round',
                              side_effect=core.StoreError('offline_db_write_failed')):
                result = service.check_branches_from_config(self.path, [branch], {})
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error']['category'], 'offline_db_write_failed')
        self.assertEqual(len(result['maintenance_api']['llm_calls']), 1)
        self.assertEqual(result['maintenance_api']['llm_calls'][0]['usage']['total_tokens'], 11)
        self.assertEqual(result['maintenance_api']['usage']['total_usage']['total_tokens'], 11)
        model.client.close.assert_called_once_with()
        self.assertEqual(constructor.call_args.kwargs['max_api_attempts'], 1)
        session_id = self.store.connection.execute('SELECT session_id FROM check_sessions').fetchone()[0]
        self.assertEqual(self.store.get_check_session(session_id)['status'], 'busy')
        self.assertEqual(self.store.get_check_session(session_id)['round_count'], 1)

    def test_model_with_hidden_retries_is_rejected_before_any_request(self):
        model = FakeModel([])
        model.public_configuration['effective']['max_total_attempts'] = 3
        with self.assertRaises(core.StoreValidation) as caught:
            service.MilestoneService(self.store, model)
        self.assertEqual(caught.exception.category, 'meta_requires_single_request_per_round')
        self.assertEqual(model.calls, [])

    def test_duplicate_json_keys_cannot_override_status_or_milestone_spec(self):
        for text in ('{"judgments":[],"judgments":[]}',
                     '{"milestones":[],"milestones":[]}',
                     '{"judgments":NaN}'):
            with self.subTest(text=text):
                with self.assertRaises(core.StoreValidation):
                    service.parse_object(text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
