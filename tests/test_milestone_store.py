"""Standard-library offline tests: frozen objectives, evidence and round budgets."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest

REPO = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
PACKAGE = 'tested_milestone_store_package'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(REPO/'hyperskill')]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE+'.'+name, REPO/'hyperskill'/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core, storage = load('evidence_store'), load('milestone_store')
PROFILE = {'model_configuration': {'model': 'offline-fixture', 'endpoint': 'https://example.invalid/v1'},
           'max_checker_rounds': 3}
LEDGER = [{'sequence': 1, 'usage': {'prompt_tokens': 5, 'completion_tokens': 6, 'total_tokens': 11}}]


def spec():
    return {'milestones': [
        {'id': 'm1', 'requirement': 'Report the fixture fact', 'target_state': 'Fact is present.',
         'evidence_rules': ['Read the captured fixture answer.'], 'dependencies': [],
         'acceptance_time': 'at_delivery', 'alternatives': [], 'weight': '0.7',
         'weight_reason': 'The primary fact is the main requested result.'},
        {'id': 'm2', 'requirement': 'Give the source', 'target_state': 'Fixture source is cited.',
         'evidence_rules': ['Read the captured answer and fixture source observation.'],
         'dependencies': [], 'acceptance_time': 'at_delivery', 'alternatives': [],
         'weight': '0.3', 'weight_reason': 'The source supports the requested result.'}]}


class SchemaTests(unittest.TestCase):
    def test_binary_spec_accepts_exact_weight_and_zero(self):
        self.assertEqual(storage.validate_spec(spec()), spec())
        value = spec(); value['milestones'][0]['weight'] = '0'; value['milestones'][1]['weight'] = '1'
        storage.validate_spec(value)

    def test_inexact_sum_is_rejected_without_rounding(self):
        for weights in [('0.7', '0.2'), ('0.7', '0.3000000000000000000000000000000000001')]:
            value = spec()
            for item, weight in zip(value['milestones'], weights): item['weight'] = weight
            with self.assertRaises(core.StoreValidation): storage.validate_spec(value)

    def test_non_decimal_partial_and_additional_fields_rejected(self):
        for weight in [0.7, True, '-0.1', 'NaN', 'Infinity', '1e0', '2', '00.7']:
            value = spec(); value['milestones'][0]['weight'] = weight
            with self.assertRaises(core.StoreValidation): storage.validate_spec(value)
        value = spec(); value['milestones'][0]['partial_bands'] = [0, 0.5, 1]
        with self.assertRaises(core.StoreValidation): storage.validate_spec(value)

    def test_duplicate_ids_missing_rules_and_invalid_time_rejected(self):
        for mutate in [lambda s: s['milestones'][1].update(id='m1'),
                       lambda s: s['milestones'][0].update(evidence_rules=[]),
                       lambda s: s['milestones'][0].update(acceptance_time='whenever')]:
            value = spec(); mutate(value)
            with self.assertRaises(core.StoreValidation): storage.validate_spec(value)

    def test_dependency_graph_must_be_existing_and_acyclic(self):
        valid = spec(); valid['milestones'][1]['dependencies'] = ['m1']; storage.validate_spec(valid)
        for references in [('m2', 'm1'), ('m1', None), ('missing', None)]:
            value = spec()
            value['milestones'][0]['dependencies'] = [references[0]]
            if references[1]: value['milestones'][1]['dependencies'] = [references[1]]
            with self.assertRaises(core.StoreValidation): storage.validate_spec(value)

    def test_strict_json_pointer_resolution(self):
        payload = {'a/b': {'~name': ['fixture', None]}}
        self.assertEqual(storage.resolve_pointer(payload, '/a~1b/~0name/0'), 'fixture')
        self.assertIsNone(storage.resolve_pointer(payload, '/a~1b/~0name/1'))
        self.assertEqual(storage.resolve_pointer(payload, ''), payload)
        for pointer in ['a/b', '/a~2b', '/a~1b/~0name/01', '/a~1b/~0name/-1', '/a~1b/~0name/9']:
            with self.assertRaises(core.StoreValidation): storage.resolve_pointer(payload, pointer)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='hyperskill-milestone-store-test-')
        self.path = Path(self.tmp.name)/'fixture.sqlite3'
        self.store = storage.MilestoneStore(self.path)
        self.protocol = self.store.register_protocol({'model': 'offline-fixture'})
        self.store.register_task('q1', 'Report the fixture fact with source.', main_branch_id='main1')

    def tearDown(self):
        self.store.close(); self.tmp.cleanup()

    def freeze(self, task_id='q1'):
        claim = self.store.claim_spec(task_id, PROFILE)
        return self.store.commit_spec(task_id, claim['owner_token'], spec(), LEDGER, json.dumps(spec()))

    def branch(self, branch_id='main1', task_id='q1', roles=None):
        frozen = self.store.get_spec(task_id)
        self.store.register_branch(branch_id, task_id, self.protocol,
            roles=roles if roles is not None else ['main'], milestone_spec_ref=frozen['spec_id'])
        evidence_id = self.store.put_evidence('result', {'final_answer': 'Fixture fact and source.',
                                                       'branch_marker': branch_id})
        self.store.attach_branch_evidence(branch_id, 'result.json', evidence_id)
        self.store.record_branch_result(branch_id, {'final_answer': 'Fixture fact and source.'})
        return evidence_id

    def session(self, multiple=False):
        self.freeze(); self.branch()
        branch_ids = ['main1']
        if multiple:
            self.branch('other1', roles=['candidate']); branch_ids.append('other1')
        return self.store.create_check_session(branch_ids, PROFILE)

    def judgments(self, session, claim, statuses=None):
        output = []
        for request in claim['requests']:
            branch_id = request['branch_id']
            evidence_id = session['snapshot'][branch_id]['evidence_refs'][0]['evidence_id']
            for identifier in request['milestone_ids']:
                status = (statuses or {}).get((branch_id, identifier), 'met')
                output.append({'branch_id': branch_id, 'milestone_id': identifier, 'status': status,
                    'reason': 'Explicit offline fixture judgment.',
                    'evidence_refs': [] if status == 'unknown' else [{'evidence_id': evidence_id, 'pointer': '/final_answer'}],
                    'unknown_reason': 'Missing fixture proof.' if status == 'unknown' else None})
        return output

    def test_spec_claim_is_concurrent_and_frozen_cache_has_no_new_rows(self):
        first = self.store.claim_spec('q1', PROFILE)
        with storage.MilestoneStore(self.path) as other:
            self.assertEqual(other.claim_spec('q1', PROFILE)['status'], 'busy')
        frozen = self.store.commit_spec('q1', first['owner_token'], spec(), LEDGER, 'fixture')
        before = self.store.summary()
        repeated = self.store.claim_spec('q1', PROFILE)
        self.assertEqual(repeated['status'], 'already_completed')
        self.assertEqual(repeated['spec_id'], frozen['spec_id'])
        self.assertEqual(self.store.summary(), before)

    def test_spec_same_commit_is_idempotent_and_changed_payload_conflicts(self):
        claim = self.store.claim_spec('q1', PROFILE)
        result = self.store.commit_spec('q1', claim['owner_token'], spec(), LEDGER, 'fixture')
        before = self.store.summary()
        again = self.store.commit_spec('q1', claim['owner_token'], spec(), LEDGER, 'fixture')
        self.assertEqual(result['spec_id'], again['spec_id']); self.assertEqual(before, self.store.summary())
        with self.assertRaises(core.StoreConflict):
            self.store.commit_spec('q1', claim['owner_token'], spec(), LEDGER, 'different raw output')

    def test_spec_failure_records_paid_usage_and_requires_explicit_retry(self):
        first = self.store.claim_spec('q1', PROFILE)
        self.store.fail_spec('q1', first['owner_token'], {'category': 'fixture_failure'}, LEDGER, 'bad json')
        self.assertEqual(self.store.claim_spec('q1', PROFILE)['status'], 'failed')
        second = self.store.claim_spec('q1', PROFILE, retry_failed=True)
        self.assertEqual(second['status'], 'claimed'); self.assertNotEqual(first['owner_token'], second['owner_token'])
        self.store.commit_spec('q1', second['owner_token'], spec(), LEDGER, 'fixture')
        events = self.store.connection.execute('SELECT evidence_id FROM meta_api_events').fetchall()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(self.store.get_evidence(row[0])['payload']['calls'] == LEDGER for row in events))

    def test_spec_rejects_wrong_owner_and_profile_change_even_after_freeze(self):
        claim = self.store.claim_spec('q1', PROFILE)
        with self.assertRaises(core.StoreConflict): self.store.commit_spec('q1', 'wrong', spec(), [], 'fixture')
        self.store.commit_spec('q1', claim['owner_token'], spec(), [], 'fixture')
        with self.assertRaises(core.StoreConflict): self.store.claim_spec('q1', {**PROFILE, 'changed': True})

    def test_historical_or_already_planned_task_cannot_be_retroactively_frozen(self):
        self.store.register_branch('main1', 'q1', self.protocol, roles=['main'])
        with self.assertRaises(core.StoreValidation): self.store.claim_spec('q1', PROFILE)
        self.store.record_branch_result('main1', {'final_answer': 'Historical fixture.'})
        with self.assertRaises(core.StoreValidation): self.store.claim_spec('q1', PROFILE)

    def test_branch_must_use_spec_and_automatic_source_reference_is_attached(self):
        frozen = self.freeze()
        for invalid in [None, 'wrong-spec']:
            with self.assertRaises(core.StoreValidation):
                self.store.register_branch('main1', 'q1', self.protocol, ['main'], milestone_spec_ref=invalid)
        self.branch()
        branch = self.store.get_branch('main1')
        self.assertEqual(branch['milestone_spec_ref'], frozen['spec_id'])
        self.assertIn(frozen['evidence_ref'], [item['evidence_id'] for item in branch['evidence_refs']])

    def test_multi_branch_session_uses_one_round_and_computes_exact_phi(self):
        session = self.session(multiple=True); claim = self.store.begin_round(session['session_id'])
        self.assertEqual(len(claim['requests']), 2)
        judgments = self.judgments(session, claim, {('main1', 'm2'): 'unmet'})
        result = self.store.commit_round(session['session_id'], claim['owner_token'], judgments, LEDGER, 'fixture')
        self.assertTrue(result['closed']); self.assertEqual(result['round_count'], 1)
        self.assertEqual(result['branches']['main1']['phi'], '0.7')
        self.assertEqual(result['branches']['other1']['phi'], '1.0')
        self.assertEqual(self.store.begin_round(session['session_id'])['status'], 'completed')

    def test_only_unknown_is_rechecked_and_unmet_is_preserved(self):
        session = self.session(); sid = session['session_id']; claim = self.store.begin_round(sid)
        judgments = self.judgments(session, claim, {('main1', 'm1'): 'unmet', ('main1', 'm2'): 'unknown'})
        first = self.store.commit_round(sid, claim['owner_token'], judgments, LEDGER, 'first')
        self.assertIsNone(first['branches']['main1']['phi'])
        second = self.store.begin_round(sid)
        self.assertEqual(second['requests'], [{'branch_id': 'main1', 'milestone_ids': ['m2']}])
        result = self.store.commit_round(sid, second['owner_token'], self.judgments(session, second), LEDGER, 'second')
        self.assertEqual(result['branches']['main1']['phi'], '0.3')
        self.assertEqual(result['branches']['main1']['judgments'][0]['status'], 'unmet')

    def test_unknown_exhaustion_is_three_shared_rounds_and_never_zero_or_renormalized(self):
        session = self.session(multiple=True); sid = session['session_id']
        for number in (1, 2, 3):
            claim = self.store.begin_round(sid); self.assertEqual(claim['round_no'], number)
            statuses = {(branch_id, 'm2'): 'unknown' for branch_id in session['branch_ids']}
            current = self.store.commit_round(sid, claim['owner_token'], self.judgments(session, claim, statuses), LEDGER, str(number))
        self.assertTrue(current['closed']); self.assertEqual(current['status'], 'unknown')
        self.assertEqual(self.store.begin_round(sid)['status'], 'exhausted')
        for state in current['branches'].values():
            self.assertIsNone(state['phi']); self.assertEqual(state['known_subtotal'], '0.7')
            self.assertEqual(state['unknown_ids'], ['m2'])
        self.assertEqual(self.store.summary()['counts']['meta_api_events'], 4)

    def test_another_session_or_profile_cannot_reset_a_branch_budget(self):
        session = self.session(multiple=True)
        self.assertEqual(self.store.create_check_session(list(reversed(session['branch_ids'])), PROFILE)['session_id'], session['session_id'])
        with self.assertRaises(core.StoreConflict): self.store.create_check_session(['main1'], PROFILE)
        with self.assertRaises(core.StoreConflict): self.store.create_check_session(session['branch_ids'], {**PROFILE, 'changed': True})

    def test_wrong_branch_or_nonexistent_pointer_evidence_is_rejected(self):
        session = self.session(multiple=True); sid = session['session_id']; claim = self.store.begin_round(sid)
        good = self.judgments(session, claim)
        bad = copy.deepcopy(good)
        bad[0]['evidence_refs'][0]['evidence_id'] = session['snapshot']['other1']['evidence_refs'][0]['evidence_id']
        with self.assertRaises(core.StoreValidation): self.store.commit_round(sid, claim['owner_token'], bad, [], 'wrongbranch')
        bad = copy.deepcopy(good); bad[0]['evidence_refs'][0]['pointer'] = '/missing'
        with self.assertRaises(core.StoreValidation): self.store.commit_round(sid, claim['owner_token'], bad, [], 'wrongpointer')
        self.assertEqual(self.store.get_check_session(sid)['status'], 'busy')
        self.store.commit_round(sid, claim['owner_token'], good, LEDGER, 'valid')

    def test_missing_duplicate_extra_or_no_evidence_judgments_are_rejected(self):
        session = self.session(); sid = session['session_id']; claim = self.store.begin_round(sid)
        good = self.judgments(session, claim)
        candidates = [good[:-1], good+[good[0]], [dict(good[0], milestone_id='invented')]+good[1:],
                      [dict(good[0], evidence_refs=[])]+good[1:]]
        for bad in candidates:
            with self.assertRaises(core.StoreValidation): self.store.commit_round(sid, claim['owner_token'], bad, [], 'bad')
        self.store.fail_round(sid, claim['owner_token'], {'category': 'invalid_fixture'}, LEDGER, 'bad')
        self.assertEqual(self.store.get_check_session(sid)['round_count'], 1)

    def test_round_failure_keeps_ledger_consumes_budget_and_is_idempotent(self):
        session = self.session(); sid = session['session_id']; claim = self.store.begin_round(sid)
        first = self.store.fail_round(sid, claim['owner_token'], {'category': 'fixture_failure'}, LEDGER, 'bad json')
        before = self.store.summary()
        again = self.store.fail_round(sid, claim['owner_token'], {'category': 'fixture_failure'}, LEDGER, 'bad json')
        self.assertEqual(first, again); self.assertEqual(before, self.store.summary())
        self.assertEqual(again['rounds'][0]['state'], 'failed'); self.assertIsNone(again['branches']['main1']['phi'])
        ledger_payload = self.store.get_evidence(again['rounds'][0]['llm_ledger_refs'][0])['payload']
        self.assertEqual(ledger_payload['calls'], LEDGER)
        self.assertEqual(self.store.begin_round(sid)['round_no'], 2)

    def test_running_round_is_not_reclaimed_on_reopen(self):
        session = self.session(); sid = session['session_id']; claim = self.store.begin_round(sid)
        with storage.MilestoneStore(self.path) as other:
            self.assertEqual(other.begin_round(sid)['status'], 'busy')
            self.assertEqual(other.get_check_session(sid)['round_count'], 1)
        self.assertEqual(claim['round_no'], 1)

    def test_same_round_commit_is_idempotent_but_changed_verdict_conflicts(self):
        session = self.session(); sid = session['session_id']; claim = self.store.begin_round(sid)
        judgments = self.judgments(session, claim)
        first = self.store.commit_round(sid, claim['owner_token'], judgments, LEDGER, 'fixture')
        before = self.store.summary()
        again = self.store.commit_round(sid, claim['owner_token'], judgments, LEDGER, 'fixture')
        self.assertEqual(first, again); self.assertEqual(before, self.store.summary())
        with self.assertRaises(core.StoreConflict):
            self.store.commit_round(sid, claim['owner_token'], judgments, LEDGER, 'different')

    def test_snapshot_excludes_maintenance_and_does_not_grow(self):
        self.freeze(); self.branch()
        unrelated = self.store.put_evidence('maintenance_fixture', {'opinion': 'Not execution evidence.'})
        self.store.attach_branch_evidence('main1', 'maintenance_fixture', unrelated)
        session = self.store.create_check_session(['main1'], PROFILE)
        refs = session['snapshot']['main1']['evidence_refs']
        self.assertEqual([ref['name'] for ref in refs], ['result.json'])
        later = self.store.put_evidence('trajectory', [{'obs': 'Later fixture.'}])
        self.store.attach_branch_evidence('main1', 'trajectory.json', later)
        self.assertEqual(self.store.get_check_session(session['session_id'])['snapshot']['main1']['evidence_refs'], refs)

    def test_session_requires_same_task_and_completed_execution(self):
        self.freeze()
        frozen = self.store.get_spec('q1')
        self.store.register_branch('main1', 'q1', self.protocol, ['main'], milestone_spec_ref=frozen['spec_id'])
        with self.assertRaises(core.StoreValidation): self.store.create_check_session(['main1'], PROFILE)
        self.store.record_branch_result('main1', {'final_answer': 'Fixture'})
        with self.assertRaises(core.StoreValidation): self.store.create_check_session(['main1'], PROFILE)
        evidence_id = self.store.put_evidence('result', {'final_answer': 'Fixture'})
        self.store.attach_branch_evidence('main1', 'result.json', evidence_id)
        self.store.register_task('q2', 'Another fixture task.', 'main2'); self.freeze('q2'); self.branch('main2', 'q2')
        with self.assertRaises(core.StoreValidation): self.store.create_check_session(['main1', 'main2'], PROFILE)

    def test_secret_material_never_commits(self):
        self.store.secrets = ('fixture-private-value',)
        with self.assertRaises(core.StoreValidation): self.store.claim_spec('q1', {'model': 'fixture-private-value'})
        self.assertEqual(self.store.summary()['counts']['milestone_jobs'], 0)

    def test_v1_and_v2_additive_migrations_preserve_every_legacy_row(self):
        for version in (1, 2):
            legacy = Path(self.tmp.name)/('legacy'+str(version)+'.sqlite3')
            tables = core.V1_TABLES + (core.LEARNING_TABLES if version == 2 else ())
            conn = sqlite3.connect(legacy)
            conn.executescript(core.V1_SCHEMA+(core.LEARNING_SCHEMA if version == 2 else ''))
            conn.execute('PRAGMA application_id='+str(core.APPLICATION_ID)); conn.execute('PRAGMA user_version='+str(version))
            conn.execute('INSERT INTO skills VALUES(?,?,?)', ('legacy-skill', '{"content":"legacy fixture"}', 'original-time'))
            conn.execute('INSERT INTO skill_versions VALUES(?,?,?,?,?,?,?)',
                ('legacy-version', 'legacy-skill', 1, 'legacy-sha', '[0.2,17]', '{"version_id":"legacy-version"}', 'original-time'))
            if version == 2:
                conn.execute('INSERT INTO embedding_profiles VALUES(?,?,?)', ('legacy-profile', '{"dimensions":3}', 'original-time'))
            before = {table: conn.execute('SELECT * FROM '+table).fetchall() for table in tables}
            conn.commit(); conn.close()
            with storage.MilestoneStore(legacy, create=False) as upgraded:
                self.assertEqual(upgraded.summary()['schema_version'], 3)
                self.assertEqual({table: [tuple(row) for row in upgraded.connection.execute('SELECT * FROM '+table)]
                                  for table in tables}, before)
                self.assertEqual(upgraded.integrity_check(), {'integrity': ['ok'], 'foreign_key_violations': []})
                self.assertTrue(all(upgraded.summary()['counts'][table] == 0 for table in core.MILESTONE_TABLES))


if __name__ == '__main__': unittest.main()
