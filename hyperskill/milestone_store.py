"""Frozen task objectives and bounded, evidence-bound Meta-Agent checks.

Only operation states are used here. Skill values, pairing, diagnosis and deletion
are outside this module. A branch is bound to one checker session so a new session
cannot silently reset its three-round budget.
"""
from decimal import Decimal, localcontext
import json
import re
import uuid

from .evidence_store import (
    EvidenceStore, StoreValidation, StoreConflict, StoreNotFound,
    canonical_json, fingerprint, _now,
)

EXECUTION_NAMES = ('result.json', 'trajectory.json', 'messages.json',
                   'tool_events.json', 'page_fetches.json', 'api_calls.json')
SPEC_KEYS = {'id', 'requirement', 'target_state', 'evidence_rules', 'dependencies',
             'acceptance_time', 'alternatives', 'weight', 'weight_reason'}
JUDGMENT_KEYS = {'branch_id', 'milestone_id', 'status', 'reason',
                 'evidence_refs', 'unknown_reason'}


def _text(value, category):
    if not isinstance(value, str) or not value.strip():
        raise StoreValidation(category)
    return value


def validate_spec(spec):
    """Validate binary objectives and exact decimal weights; never normalize."""
    canonical_json(spec)
    if (not isinstance(spec, dict) or set(spec) != {'milestones'}
            or not isinstance(spec['milestones'], list) or not spec['milestones']):
        raise StoreValidation('invalid_milestone_spec')
    identifiers = set()
    weights = []
    for item in spec['milestones']:
        if not isinstance(item, dict) or set(item) != SPEC_KEYS:
            raise StoreValidation('invalid_milestone_fields')
        for key in ('id', 'requirement', 'target_state', 'weight_reason'):
            _text(item[key], 'milestone_text_required')
        if item['id'] in identifiers:
            raise StoreValidation('duplicate_milestone_id')
        identifiers.add(item['id'])
        for key in ('evidence_rules', 'dependencies', 'alternatives'):
            if (not isinstance(item[key], list)
                    or any(not isinstance(v, str) or not v.strip() for v in item[key])
                    or len(set(item[key])) != len(item[key])):
                raise StoreValidation('invalid_milestone_list')
        if not item['evidence_rules']:
            raise StoreValidation('milestone_evidence_rules_required')
        if item['acceptance_time'] not in ('at_delivery', 'during_execution'):
            raise StoreValidation('invalid_milestone_acceptance_time')
        weight = item['weight']
        if (not isinstance(weight, str)
                or not re.fullmatch(r'(?:0|1)(?:\.[0-9]+)?', weight)
                or len(weight) > 80):
            raise StoreValidation('decimal_milestone_weight_required')
        value = Decimal(weight)
        if not Decimal(0) <= value <= Decimal(1):
            raise StoreValidation('milestone_weight_out_of_range')
        weights.append(weight)
    if _weight_sum(weights) != Decimal(1):
        raise StoreValidation('milestone_weights_must_sum_to_one')
    dependencies = {item['id']: item['dependencies'] for item in spec['milestones']}
    if any(ref not in identifiers for refs in dependencies.values() for ref in refs):
        raise StoreValidation('unknown_milestone_dependency')
    visited, active = set(), set()
    def visit(identifier):
        if identifier in active:
            raise StoreValidation('cyclic_milestone_dependencies')
        if identifier in visited:
            return
        active.add(identifier)
        for child in dependencies[identifier]:
            visit(child)
        active.remove(identifier)
        visited.add(identifier)
    try:
        for identifier in dependencies:
            visit(identifier)
    except RecursionError:
        raise StoreValidation('milestone_dependencies_too_deep') from None
    return spec


def resolve_pointer(payload, pointer):
    """Resolve strict RFC 6901 JSON pointers relative to an evidence payload."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise StoreValidation('invalid_evidence_pointer')
    if pointer == '':
        return payload
    current = payload
    for part in pointer[1:].split('/'):
        if re.search(r'~(?:[^01]|$)', part):
            raise StoreValidation('invalid_evidence_pointer')
        key = part.replace('~1', '/').replace('~0', '~')
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and re.fullmatch(r'0|[1-9][0-9]*', key):
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                raise StoreValidation('evidence_pointer_not_found') from None
        else:
            raise StoreValidation('evidence_pointer_not_found')
    return current


def _decimal_text(value):
    return format(value, 'f')


def _weight_sum(weights):
    # Decimal's ambient precision defaults to 28. Choose enough precision for
    # every supplied digit so near-one weights cannot round into acceptance.
    with localcontext() as context:
        context.prec = max(100, sum(len(weight) for weight in weights)+10)
        return sum((Decimal(weight) for weight in weights), Decimal(0))


class MilestoneStore(EvidenceStore):
    def _before_branches(self, task_id):
        self._require('tasks', 'task_id', task_id)
        if self.connection.execute('SELECT 1 FROM branches WHERE task_id=?', (task_id,)).fetchone():
            raise StoreValidation('milestones_must_precede_all_branches')

    def _spec_job(self, task_id):
        return self._require('milestone_jobs', 'task_id', task_id)

    def _spec_owner(self, task_id, owner):
        row = self._spec_job(task_id)
        if row['state'] != 'running' or row['owner_token'] != owner:
            raise StoreConflict('milestone_job_not_owned_running')
        return row

    def get_spec(self, task_id):
        self._id(task_id)
        row = self._require('milestone_specs', 'task_id', task_id)
        return {**json.loads(row['data_json']), 'evidence_ref': row['evidence_id'],
                'created_at': row['created_at']}

    def get_spec_job(self, task_id):
        row = self._spec_job(task_id)
        data = {'task_id': task_id, 'state': row['state'],
                'profile': json.loads(row['data_json'])['profile'],
                'created_at': row['created_at'], 'updated_at': row['updated_at']}
        for field in ('receipt_evidence_id', 'error_evidence_id'):
            data[field] = row[field]
        return data

    def claim_spec(self, task_id, profile, retry_failed=False):
        self._id(task_id)
        if not isinstance(profile, dict):
            raise StoreValidation('milestone_profile_required')
        self._validate(profile)
        data = {'task_id': task_id, 'profile': profile}
        with self.transaction():
            row = self.connection.execute('SELECT * FROM milestone_jobs WHERE task_id=?', (task_id,)).fetchone()
            if row:
                if row['data_json'] != canonical_json(data):
                    raise StoreConflict('milestone_profile_conflict')
                if row['state'] == 'completed':
                    return {'status': 'already_completed', **self.get_spec(task_id)}
                if row['state'] == 'running':
                    return {'status': 'busy', **self.get_spec_job(task_id)}
                if not retry_failed:
                    return {'status': 'failed', **self.get_spec_job(task_id)}
            self._before_branches(task_id)
            owner = uuid.uuid4().hex
            if row:
                self.connection.execute("UPDATE milestone_jobs SET state='running',owner_token=?,updated_at=? WHERE task_id=?",
                                        (owner, _now(), task_id))
            else:
                self._put('milestone_jobs', 'task_id', task_id, data,
                          {'state': 'running', 'owner_token': owner, 'updated_at': _now()})
            return {'status': 'claimed', 'owner_token': owner, **self.get_spec_job(task_id)}

    def _log_api(self, operation_id, kind, owner, ledger, round_no=None):
        if not isinstance(ledger, list) or any(not isinstance(row, dict) for row in ledger):
            raise StoreValidation('invalid_meta_api_ledger')
        if not ledger:
            return []
        payload = {'operation_id': operation_id, 'kind': kind, 'owner': owner,
                   'round_no': round_no, 'calls': ledger}
        self._validate(payload)
        evidence_id = self.put_evidence('meta_agent_api_calls', payload)
        event_id = fingerprint(payload)
        self._put('meta_api_events', 'event_id', event_id,
                  {'operation_id': operation_id, 'kind': kind, 'evidence_id': evidence_id},
                  {'operation_id': operation_id, 'kind': kind, 'evidence_id': evidence_id})
        return [evidence_id]

    def commit_spec(self, task_id, owner_token, spec, llm_ledger, raw_response):
        validate_spec(spec)
        self._validate({'spec': spec, 'llm_ledger': llm_ledger, 'raw_response': raw_response})
        with self.transaction():
            row = self._spec_job(task_id)
            if row['state'] == 'completed' and row['owner_token'] == owner_token:
                frozen = self.get_spec(task_id)
                original = self.get_evidence(frozen['evidence_ref'])['payload']
                if (frozen['spec'] != spec or original['raw_response'] != raw_response
                        or not self._ledger_matches(original['llm_ledger_refs'], llm_ledger)):
                    raise StoreConflict('frozen_milestone_spec_conflict')
                return {'status': 'already_completed', **frozen}
            row = self._spec_owner(task_id, owner_token)
            self._before_branches(task_id)
            profile = json.loads(row['data_json'])['profile']
            spec_id = 'ms_'+fingerprint({'task_id': task_id, 'spec': spec, 'profile': profile})
            refs = self._log_api(task_id, 'milestone_generation', owner_token, llm_ledger)
            payload = {'task_id': task_id, 'spec_id': spec_id, 'spec': spec,
                       'profile': profile, 'raw_response': raw_response, 'llm_ledger_refs': refs,
                       'frozen_before_any_branch': True}
            evidence_id = self.put_evidence('frozen_milestone_spec', payload)
            self._put('milestone_specs', 'spec_id', spec_id,
                      {'task_id': task_id, 'spec_id': spec_id, 'spec': spec,
                       'profile': profile, 'llm_ledger_refs': refs,
                       'frozen_before_any_branch': True},
                      {'task_id': task_id, 'evidence_id': evidence_id})
            self.connection.execute("UPDATE milestone_jobs SET state='completed',receipt_evidence_id=?,updated_at=? WHERE task_id=?",
                                    (evidence_id, _now(), task_id))
            return {'status': 'completed', **self.get_spec(task_id)}

    def fail_spec(self, task_id, owner_token, error, llm_ledger, raw_response=None):
        self._validate({'error': error, 'raw_response': raw_response})
        with self.transaction():
            self._spec_owner(task_id, owner_token)
            refs = self._log_api(task_id, 'milestone_generation', owner_token, llm_ledger)
            evidence_id = self.put_evidence('milestone_generation_error',
                {'task_id': task_id, 'owner': owner_token, 'error': error,
                 'raw_response': raw_response, 'llm_ledger_refs': refs})
            self.connection.execute("UPDATE milestone_jobs SET state='failed',error_evidence_id=?,updated_at=? WHERE task_id=?",
                                    (evidence_id, _now(), task_id))
            return {'status': 'failed', **self.get_spec_job(task_id)}

    def create_check_session(self, branch_ids, profile):
        if (not isinstance(branch_ids, list) or not branch_ids
                or any(not isinstance(v, str) or not v.strip() for v in branch_ids)
                or len(set(branch_ids)) != len(branch_ids)):
            raise StoreValidation('checker_branches_required')
        if not isinstance(profile, dict):
            raise StoreValidation('checker_profile_required')
        self._validate(profile)
        branch_ids = sorted(branch_ids)
        with self.transaction():
            bindings = [r[0] for branch_id in branch_ids for r in self.connection.execute(
                'SELECT session_id FROM check_bindings WHERE branch_id=?', (branch_id,))]
            if bindings:
                if len(bindings) != len(branch_ids) or len(set(bindings)) != 1:
                    raise StoreConflict('branch_check_budget_already_bound')
                previous = self.get_check_session(bindings[0])
                if previous['branch_ids'] != branch_ids or previous['profile'] != profile:
                    raise StoreConflict('branch_check_budget_already_bound')
                return {'status': 'already_exists', **previous}
            snapshot, task_id, frozen = {}, None, None
            for branch_id in branch_ids:
                branch = self.get_branch(branch_id)
                if branch['result'] is None:
                    raise StoreValidation('completed_branch_result_required')
                if task_id is None:
                    task_id = branch['task_id']
                    frozen = self.get_spec(task_id)
                if branch['task_id'] != task_id:
                    raise StoreValidation('checker_branches_must_share_task')
                if branch['milestone_spec_ref'] != frozen['spec_id']:
                    raise StoreValidation('checker_branch_spec_mismatch')
                refs = [ref for ref in branch['evidence_refs'] if ref['name'] in EXECUTION_NAMES]
                if not refs:
                    raise StoreValidation('checker_execution_evidence_required')
                snapshot[branch_id] = {key: branch[key] for key in (
                    'branch_id', 'protocol_id', 'roles', 'supplied_versions',
                    'target_version_id', 'initial_state_ref')}
                snapshot[branch_id]['evidence_refs'] = refs
            data = {'task_id': task_id, 'spec_id': frozen['spec_id'], 'branch_ids': branch_ids,
                    'profile': profile, 'snapshot': snapshot, 'max_rounds': 3,
                    'binary_milestones_only': True, 'evidence_mode': 'frozen_execution_evidence'}
            session_id = 'check_'+fingerprint(data)
            self._put('check_sessions', 'session_id', session_id, data,
                      {'task_id': task_id, 'spec_id': frozen['spec_id'],
                       'state': 'active', 'updated_at': _now()})
            self.connection.executemany('INSERT INTO check_bindings VALUES(?,?)',
                                       [(branch_id, session_id) for branch_id in branch_ids])
            return {'status': 'created', **self.get_check_session(session_id)}

    def _session_row(self, session_id):
        self._id(session_id)
        return self._require('check_sessions', 'session_id', session_id)

    def _round_owner(self, session_id, owner):
        self._session_row(session_id)
        row = self.connection.execute('SELECT * FROM check_rounds WHERE session_id=? AND owner_token=?',
                                      (session_id, owner)).fetchone()
        if row is None or row['state'] != 'running':
            raise StoreConflict('checker_round_not_owned_running')
        return row

    def _ledger_matches(self, refs, ledger):
        if not refs:
            return ledger == []
        return (len(refs) == 1
                and self.get_evidence(refs[0])['payload']['calls'] == ledger)

    def get_check_session(self, session_id):
        row = self._session_row(session_id)
        data = json.loads(row['data_json'])
        spec = self.get_spec(data['task_id'])['spec']
        states = {branch_id: {item['id']: {'branch_id': branch_id, 'milestone_id': item['id'],
                     'status': 'unknown', 'reason': 'Not checked yet.', 'evidence_refs': [],
                     'unknown_reason': 'not_checked'} for item in spec['milestones']}
                  for branch_id in data['branch_ids']}
        rounds = []
        for record in self.connection.execute('SELECT * FROM check_rounds WHERE session_id=? ORDER BY round_no', (session_id,)):
            value = json.loads(record['data_json'])
            for judgment in value.get('judgments', []):
                states[judgment['branch_id']][judgment['milestone_id']] = judgment
            rounds.append({'round_no': record['round_no'], 'state': record['state'],
                           'requests': json.loads(record['request_json']),
                           'evidence_ref': record['evidence_id'],
                           'created_at': record['created_at'], 'updated_at': record['updated_at'], **value})
        branches = {}
        for branch_id, items in states.items():
            unknown = [item['id'] for item in spec['milestones'] if items[item['id']]['status'] == 'unknown']
            subtotal = _weight_sum([item['weight'] for item in spec['milestones']
                                   if items[item['id']]['status'] == 'met'])
            branches[branch_id] = {'judgments': [items[item['id']] for item in spec['milestones']],
                 'phi': None if unknown else _decimal_text(subtotal),
                 'known_subtotal': _decimal_text(subtotal), 'unknown_ids': unknown}
        has_unknown = any(branch['unknown_ids'] for branch in branches.values())
        running = any(item['state'] == 'running' for item in rounds)
        closed = row['state'] == 'closed'
        status = 'busy' if running else 'unknown' if closed and has_unknown else 'completed' if closed else 'ready'
        event_refs = [r[0] for r in self.connection.execute(
            'SELECT evidence_id FROM meta_api_events WHERE operation_id=? ORDER BY created_at,event_id', (session_id,))]
        return {**data, 'session_id': session_id, 'round_count': len(rounds),
                'closed': closed, 'status': status, 'rounds': rounds, 'branches': branches,
                'llm_ledger_refs': event_refs, 'created_at': row['created_at'], 'updated_at': row['updated_at']}

    def begin_round(self, session_id):
        with self.transaction():
            session = self.get_check_session(session_id)
            if session['status'] == 'busy':
                return {'status': 'busy', 'session_id': session_id}
            if session['closed']:
                return {'status': 'completed' if session['status'] == 'completed' else 'exhausted',
                        'session_id': session_id}
            requests = [{'branch_id': branch_id, 'milestone_ids': state['unknown_ids']}
                        for branch_id, state in session['branches'].items() if state['unknown_ids']]
            if session['round_count'] >= 3 or not requests:
                self.connection.execute("UPDATE check_sessions SET state='closed',updated_at=? WHERE session_id=?", (_now(), session_id))
                return {'status': 'exhausted' if requests else 'completed', 'session_id': session_id}
            owner, round_no, now = uuid.uuid4().hex, session['round_count']+1, _now()
            self.connection.execute('INSERT INTO check_rounds(session_id,round_no,owner_token,state,request_json,data_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',
                (session_id, round_no, owner, 'running', canonical_json(requests), '{}', now, now))
            self.connection.execute('UPDATE check_sessions SET updated_at=? WHERE session_id=?', (now, session_id))
            return {'status': 'claimed', 'session_id': session_id, 'round_no': round_no,
                    'owner_token': owner, 'requests': requests}

    def _validate_judgments(self, session, requests, judgments):
        if not isinstance(judgments, list):
            raise StoreValidation('checker_judgments_required')
        expected = {(request['branch_id'], identifier) for request in requests for identifier in request['milestone_ids']}
        actual = set()
        for item in judgments:
            if not isinstance(item, dict) or set(item) != JUDGMENT_KEYS:
                raise StoreValidation('invalid_checker_judgment_fields')
            key = (item['branch_id'], item['milestone_id'])
            if any(not isinstance(v, str) for v in key) or key not in expected or key in actual:
                raise StoreValidation('checker_judgment_scope_mismatch')
            actual.add(key)
            if item['status'] not in ('met', 'unmet', 'unknown'):
                raise StoreValidation('invalid_checker_status')
            _text(item['reason'], 'checker_reason_required')
            if item['status'] == 'unknown':
                _text(item['unknown_reason'], 'checker_unknown_reason_required')
            elif item['unknown_reason'] is not None:
                raise StoreValidation('known_checker_result_has_unknown_reason')
            refs = item['evidence_refs']
            if not isinstance(refs, list) or (item['status'] != 'unknown' and not refs):
                raise StoreValidation('checker_evidence_refs_required')
            allowed = {ref['evidence_id'] for ref in session['snapshot'][item['branch_id']]['evidence_refs']}
            for ref in refs:
                if not isinstance(ref, dict) or set(ref) != {'evidence_id', 'pointer'}:
                    raise StoreValidation('invalid_checker_evidence_ref')
                if not isinstance(ref['evidence_id'], str) or ref['evidence_id'] not in allowed:
                    raise StoreValidation('checker_evidence_wrong_branch')
                resolve_pointer(self.get_evidence(ref['evidence_id'])['payload'], ref['pointer'])
        if actual != expected:
            raise StoreValidation('checker_judgment_scope_mismatch')
        self._validate(judgments)
        return judgments

    def _finish_round(self, session_id, row, state, payload, ledger, raw_response):
        refs = self._log_api(session_id, 'milestone_check', row['owner_token'], ledger, row['round_no'])
        data = {**payload, 'raw_response': raw_response, 'llm_ledger_refs': refs}
        evidence_id = self.put_evidence('milestone_check_round',
            {'session_id': session_id, 'round_no': row['round_no'], 'requests': json.loads(row['request_json']),
             'state': state, **data})
        self.connection.execute('UPDATE check_rounds SET state=?,evidence_id=?,data_json=?,updated_at=? WHERE session_id=? AND round_no=?',
            (state, evidence_id, canonical_json(data), _now(), session_id, row['round_no']))
        session = self.get_check_session(session_id)
        for branch_id in session['branch_ids']:
            self.attach_branch_evidence(branch_id, session_id+'_round_'+str(row['round_no']), evidence_id)
        closed = row['round_no'] >= 3 or not any(item['unknown_ids'] for item in session['branches'].values())
        self.connection.execute('UPDATE check_sessions SET state=?,updated_at=? WHERE session_id=?',
                                ('closed' if closed else 'active', _now(), session_id))
        return self.get_check_session(session_id)

    def commit_round(self, session_id, owner_token, judgments, llm_ledger, raw_response):
        self._validate({'judgments': judgments, 'raw_response': raw_response})
        with self.transaction():
            previous = self.connection.execute('SELECT * FROM check_rounds WHERE session_id=? AND owner_token=?',
                                               (session_id, owner_token)).fetchone()
            if previous is not None and previous['state'] != 'running':
                value = json.loads(previous['data_json'])
                if (previous['state'] == 'completed' and value.get('judgments') == judgments
                        and value.get('raw_response') == raw_response
                        and self._ledger_matches(value['llm_ledger_refs'], llm_ledger)):
                    return self.get_check_session(session_id)
                raise StoreConflict('checker_round_already_finished')
            row = self._round_owner(session_id, owner_token)
            session = self.get_check_session(session_id)
            self._validate_judgments(session, json.loads(row['request_json']), judgments)
            return self._finish_round(session_id, row, 'completed', {'judgments': judgments}, llm_ledger, raw_response)

    def fail_round(self, session_id, owner_token, error, llm_ledger, raw_response=None):
        self._validate({'error': error, 'raw_response': raw_response})
        with self.transaction():
            previous = self.connection.execute('SELECT * FROM check_rounds WHERE session_id=? AND owner_token=?',
                                               (session_id, owner_token)).fetchone()
            if previous is not None and previous['state'] != 'running':
                value = json.loads(previous['data_json'])
                if (previous['state'] == 'failed' and value.get('error') == error
                        and value.get('raw_response') == raw_response
                        and self._ledger_matches(value['llm_ledger_refs'], llm_ledger)):
                    return self.get_check_session(session_id)
                raise StoreConflict('checker_round_already_finished')
            row = self._round_owner(session_id, owner_token)
            judgments = [{'branch_id': request['branch_id'], 'milestone_id': identifier,
                'status': 'unknown', 'reason': 'The checker round did not produce a valid judgment.',
                'unknown_reason': 'checker_round_error', 'evidence_refs': []}
                for request in json.loads(row['request_json']) for identifier in request['milestone_ids']]
            return self._finish_round(session_id, row, 'failed', {'judgments': judgments, 'error': error}, llm_ledger, raw_response)
