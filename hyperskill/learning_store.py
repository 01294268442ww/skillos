"""Routine extraction jobs and the two-route HyperSkill hypergraph.

Skills are exact versions; graph membership records source relationships only.
The job states below describe one operation, never a skill lifecycle or value.
"""
import json
import math
import uuid

from .evidence_store import (
    EvidenceStore, StoreValidation, StoreConflict, StoreNotFound,
    canonical_json, fingerprint, _now,
)


def subtask_text(subtask):
    return subtask['name'].strip()+'\n'+subtask['description'].strip()


def episode_text(task, fragment):
    return ' '.join(value.strip() for value in (task, fragment) if value.strip())


def plan_text(descriptions):
    return ' '.join(value.strip() for value in descriptions if value.strip())


def required_texts(task, extraction):
    return sorted(set([subtask_text(s) for s in extraction['subtasks']]
                      + [episode_text(task, extraction['fragment'])]))


def _profile(profile):
    if not isinstance(profile, dict):
        raise StoreValidation('embedding_profile_required')
    if any(not isinstance(profile.get(k), str) or not profile[k].strip()
           for k in ('model', 'endpoint')):
        raise StoreValidation('explicit_embedding_profile_required')
    if type(profile.get('dimensions')) is not int or profile['dimensions'] < 1:
        raise StoreValidation('invalid_embedding_dimensions')
    return profile['dimensions']


def _vector(vector, dimensions):
    if (not isinstance(vector, (list, tuple)) or len(vector) != dimensions
            or any(type(v) not in (int, float) for v in vector)):
        raise StoreValidation('invalid_embedding_vector')
    try:
        result = [float(v) for v in vector]
    except (OverflowError,ValueError):
        raise StoreValidation('invalid_embedding_vector') from None
    if any(not math.isfinite(v) for v in result):
        raise StoreValidation('invalid_embedding_vector')
    if not any(result):
        raise StoreValidation('zero_embedding_vector')
    return result


def _unit(vector):
    scale = max(abs(v) for v in vector)
    scaled = [v/scale for v in vector]
    norm = math.sqrt(sum(v*v for v in scaled))
    return [v/norm for v in scaled]


def _cosine(first, second):
    return max(-1.0,min(1.0,sum(a*b for a,b in zip(_unit(first), _unit(second)))))


class LearningStore(EvidenceStore):
    def register_embedding_profile(self, profile):
        _profile(profile)
        self._validate(profile)
        identifier = fingerprint(profile)
        with self.transaction():
            self._put('embedding_profiles', 'profile_id', identifier, profile)
        return identifier

    def _profile_data(self, profile_id):
        self._id(profile_id)
        return json.loads(self._require('embedding_profiles', 'profile_id', profile_id)['data_json'])

    def get_cached(self, profile_id, text):
        self._profile_data(profile_id); self._id(text)
        key = fingerprint({'profile_id': profile_id, 'text': text})
        row = self.connection.execute('SELECT data_json FROM embedding_cache WHERE cache_id=?', (key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return _vector(data['vector'], _profile(self._profile_data(profile_id)))

    def put_cached(self, profile_id, text, vector):
        profile = self._profile_data(profile_id); self._id(text)
        values = _vector(vector, _profile(profile))
        data = {'profile_id': profile_id, 'text': text, 'vector': values}
        key = fingerprint({'profile_id': profile_id, 'text': text})
        with self.transaction():
            self._put('embedding_cache', 'cache_id', key, data,
                      {'profile_id': profile_id, 'text_sha256': fingerprint(text)})
        return key

    def _job_row(self, branch_id):
        self._id(branch_id)
        return self._require('learning_jobs', 'branch_id', branch_id)

    def _owner(self, branch_id, owner_token):
        row = self._job_row(branch_id)
        if row['state'] != 'running' or row['owner_token'] != owner_token:
            raise StoreConflict('learning_job_not_owned_running')
        return row

    def _source_trajectory(self, branch):
        trajectory_id = next((e['evidence_id'] for e in branch['evidence_refs']
                              if e['name']=='trajectory.json'),None)
        if not trajectory_id:
            raise StoreValidation('source_trajectory_required')
        trajectory = self.get_evidence(trajectory_id)['payload']
        if not isinstance(trajectory,list):
            raise StoreValidation('invalid_source_trajectory')
        return trajectory_id,trajectory

    def get_job(self, branch_id):
        row = self._job_row(branch_id)
        events = list(self.connection.execute(
            'SELECT kind,evidence_id FROM learning_api_events WHERE job_id=? ORDER BY created_at,event_id',
            (row['job_id'],)))
        return {'job_id': row['job_id'], 'branch_id': branch_id, 'state': row['state'],
                'checkpoint': self.get_evidence(row['checkpoint_evidence_id'])['payload']['extraction']
                    if row['checkpoint_evidence_id'] else None,
                'receipt': self.get_evidence(row['receipt_evidence_id'])['payload']
                    if row['receipt_evidence_id'] else None,
                'error': self.get_evidence(row['error_evidence_id'])['payload']
                    if row['error_evidence_id'] else None,
                'llm_ledger_refs': [r['evidence_id'] for r in events if r['kind']=='llm'],
                'embedding_ledger_refs': [r['evidence_id'] for r in events if r['kind']=='embedding'],
                'configuration': json.loads(row['data_json'])}

    def claim(self, branch_id, extractor_profile, embedding_profile, retry_failed=False):
        with self.transaction():
            branch = self.get_branch(branch_id)
            if 'main' not in branch['roles']:
                return {'status': 'skipped_extra', 'branch_id': branch_id}
            if branch['result'] is None:
                raise StoreValidation('completed_branch_result_required')
            self._source_trajectory(branch)
            self._validate(extractor_profile)
            if not isinstance(extractor_profile, dict):
                raise StoreValidation('extractor_profile_required')
            _profile(embedding_profile); self._validate(embedding_profile)
            row = self.connection.execute('SELECT * FROM learning_jobs WHERE branch_id=?', (branch_id,)).fetchone()
            data = {'branch_id': branch_id, 'extractor_profile': extractor_profile,
                    'embedding_profile': embedding_profile}
            if row:
                if row['state']=='completed':
                    return {'status': 'already_completed', **self.get_job(branch_id)}
                if row['data_json'] != canonical_json(data):
                    raise StoreConflict('learning_job_profile_conflict')
                if row['state']=='running':
                    return {'status': 'busy', **self.get_job(branch_id)}
                if not retry_failed:
                    return {'status': 'failed', **self.get_job(branch_id)}
                owner = uuid.uuid4().hex
                self.connection.execute("UPDATE learning_jobs SET state='running',owner_token=?,updated_at=? WHERE branch_id=?",
                                        (owner,_now(),branch_id))
            else:
                owner = uuid.uuid4().hex
                job_id = 'learn_'+fingerprint({'branch_id':branch_id})
                self.register_embedding_profile(embedding_profile)
                self._put('learning_jobs','branch_id',branch_id,data,
                          {'job_id':job_id,'owner_token':owner,'state':'running','updated_at':_now()})
            return {'status':'claimed','owner_token':owner,**self.get_job(branch_id)}

    def log_api_events(self, job_id, kind, ledger):
        if kind not in ('llm','embedding') or not isinstance(ledger,list):
            raise StoreValidation('invalid_maintenance_ledger')
        if not ledger:
            return []
        self._validate(ledger)
        with self.transaction():
            row = self._require('learning_jobs','job_id',job_id)
            payload = {'job_id':job_id, 'owner':row['owner_token'], 'kind':kind,'calls':ledger}
            evidence_id = self.put_evidence('maintenance_api_calls',payload)
            event_id = fingerprint(payload)
            self._put('learning_api_events','event_id',event_id,
                      {'job_id':job_id,'kind':kind,'evidence_id':evidence_id},
                      {'job_id':job_id,'kind':kind,'evidence_id':evidence_id})
            self.attach_branch_evidence(row['branch_id'],'maintenance_'+event_id,evidence_id)
        return [evidence_id]

    def _validate_extraction(self, branch_id, extraction):
        self._validate(extraction)
        if (not isinstance(extraction,dict) or not isinstance(extraction.get('subtasks'),list)
                or not isinstance(extraction.get('skills'),list) or not isinstance(extraction.get('fragment'),str)):
            raise StoreValidation('invalid_extraction_schema')
        branch = self.get_branch(branch_id)
        trajectory_id,trajectory = self._source_trajectory(branch)
        for subtask in extraction['subtasks']:
            if (not isinstance(subtask,dict) or any(not isinstance(subtask.get(k),str)
                    or not subtask[k].strip() for k in ('name','description'))):
                raise StoreValidation('invalid_extracted_subtask')
        goals = {subtask['description'] for subtask in extraction['subtasks']}
        for skill in extraction['skills']:
            if (not isinstance(skill,dict) or any(not isinstance(skill.get(k),str)
                    or not skill[k].strip() for k in ('name','content'))
                    or 'conditions' not in skill or skill['conditions'] is None
                    or not isinstance(skill.get('goal_refs'),list) or not skill['goal_refs']
                    or any(not isinstance(x,str) or x not in goals for x in skill['goal_refs'])
                    or not isinstance(skill.get('evidence_steps'),list) or not skill['evidence_steps']
                    or any(type(x) is not int or x < 1 or x > len(trajectory) for x in skill['evidence_steps'])):
                raise StoreValidation('invalid_extracted_skill')
        return branch,trajectory_id

    def checkpoint(self, branch_id, owner_token, extraction, llm_ledger):
        with self.transaction():
            row = self._owner(branch_id,owner_token)
            self._validate_extraction(branch_id,extraction)
            evidence_id = self.put_evidence('routine_extraction',{'branch_id':branch_id,'extraction':extraction})
            if row['checkpoint_evidence_id'] and row['checkpoint_evidence_id'] != evidence_id:
                raise StoreConflict('extraction_checkpoint_conflict')
            self.attach_branch_evidence(branch_id,'routine_extraction',evidence_id)
            self.log_api_events(row['job_id'],'llm',llm_ledger)
            self.connection.execute('UPDATE learning_jobs SET checkpoint_evidence_id=?,updated_at=? WHERE branch_id=?',
                                    (evidence_id,_now(),branch_id))
        return evidence_id

    def fail(self, branch_id, owner_token, error, llm_ledger=None, embedding_ledger=None):
        if not isinstance(error,dict) or set(error)-{'type','category','http_status'}:
            raise StoreValidation('controlled_error_fields_required')
        self._validate(error)
        with self.transaction():
            row = self._owner(branch_id,owner_token)
            self.log_api_events(row['job_id'],'llm',llm_ledger or [])
            self.log_api_events(row['job_id'],'embedding',embedding_ledger or [])
            evidence_id = self.put_evidence('learning_error',
                {'job_id':row['job_id'],'branch_id':branch_id,'owner':row['owner_token'],'error':error})
            self.attach_branch_evidence(branch_id,'learning_error_'+evidence_id,evidence_id)
            self.connection.execute("UPDATE learning_jobs SET state='failed',error_evidence_id=?,updated_at=? WHERE branch_id=?",
                                    (evidence_id,_now(),branch_id))
        return self.get_job(branch_id)

    def commit_extraction(self, branch_id, owner_token, extraction, vectors, embedding_profile, embedding_ledger):
        with self.transaction():
            row = self._owner(branch_id,owner_token)
            config = json.loads(row['data_json'])
            if config['embedding_profile'] != embedding_profile:
                raise StoreConflict('learning_job_profile_conflict')
            branch,trajectory_id = self._validate_extraction(branch_id,extraction)
            if not row['checkpoint_evidence_id'] or self.get_job(branch_id)['checkpoint'] != extraction:
                raise StoreValidation('matching_extraction_checkpoint_required')
            profile_id = self.register_embedding_profile(embedding_profile)
            task = self.get_task(branch['task_id'])['task']
            if not isinstance(vectors,dict):
                raise StoreValidation('embedding_vectors_required')
            cache_ids = {}
            for text in required_texts(task,extraction):
                vector = vectors.get(text)
                if vector is None:
                    vector = self.get_cached(profile_id,text)
                if vector is None:
                    raise StoreValidation('missing_index_embedding')
                cache_ids[text] = self.put_cached(profile_id,text,vector)
            subtask_ids,version_ids,skill_ids = [],[],[]
            for subtask in extraction['subtasks']:
                text = subtask_text(subtask)
                node_id = 'sub_'+fingerprint({'name':subtask['name'],'description':subtask['description']})
                self._put('graph_nodes','node_id',node_id,
                          {'node_id':node_id,'kind':'subtask','name':subtask['name'],
                           'description':subtask['description'],'text':text}, {'kind':'subtask','version_id':None})
                subtask_ids.append(node_id)
            for skill in extraction['skills']:
                identity = {k:skill[k] for k in ('content','conditions','goal_refs')}
                skill_id = 'skill_'+fingerprint(identity)
                self.put_skill(skill_id)
                version_id = self.put_skill_version(skill_id,1,skill['content'],skill['conditions'],
                                                    skill['goal_refs'],'extracted')
                self.link_version_evidence(version_id,'source',branch['task_id'],branch_id,trajectory_id)
                self.link_version_evidence(version_id,'source',branch['task_id'],branch_id,row['checkpoint_evidence_id'])
                self._put('graph_nodes','node_id',version_id,
                          {'node_id':version_id,'kind':'skill','version_id':version_id},
                          {'kind':'skill','version_id':version_id})
                skill_ids.append(skill_id);version_ids.append(version_id)
            edge_id = 'he_'+fingerprint({'branch_id':branch_id})
            text = episode_text(task,extraction['fragment'])
            self._put('graph_hyperedges','hyperedge_id',edge_id,
                      {'hyperedge_id':edge_id,'branch_id':branch_id,'task_id':branch['task_id'],
                       'task':task,'fragment':extraction['fragment'],'text':text,'profile_id':profile_id},
                      {'branch_id':branch_id,'profile_id':profile_id,'cache_id':cache_ids[text]})
            self.connection.executemany('INSERT INTO graph_memberships VALUES(?,?)',
                                       [(edge_id,node) for node in sorted(set(subtask_ids+version_ids))])
            self.log_api_events(row['job_id'],'embedding',embedding_ledger)
            job = self.get_job(branch_id)
            receipt = {'status':'completed','job_id':row['job_id'],'branch_id':branch_id,
                       'skill_ids':sorted(set(skill_ids)),'version_ids':sorted(set(version_ids)),
                       'subtask_ids':sorted(set(subtask_ids)),'hyperedge_id':edge_id,
                       'source_evidence_id':trajectory_id,'extraction_evidence_id':row['checkpoint_evidence_id'],
                       'embedding_profile_id':profile_id,'llm_ledger_refs':job['llm_ledger_refs'],
                       'embedding_ledger_refs':job['embedding_ledger_refs']}
            receipt_id = self.put_evidence('learning_receipt',receipt)
            self.attach_branch_evidence(branch_id,'routine_learning_receipt',receipt_id)
            self.connection.execute("UPDATE learning_jobs SET state='completed',receipt_evidence_id=?,updated_at=? WHERE branch_id=?",
                                    (receipt_id,_now(),branch_id))
        return receipt

    def recall(self, task_query, subtask_descriptions, vectors, embedding_profile, k):
        if type(k) is not int or k < 1:
            raise StoreValidation('positive_recall_budget_required')
        self._id(task_query)
        if not isinstance(subtask_descriptions,list) or any(not isinstance(x,str) or not x.strip() for x in subtask_descriptions):
            raise StoreValidation('invalid_subtask_plan')
        dim = _profile(embedding_profile);self._validate(embedding_profile)
        profile_id = fingerprint(embedding_profile)
        if not isinstance(vectors,dict):
            raise StoreValidation('embedding_vectors_required')
        k_e,k_u = (k+1)//2,k//2
        if k_u and not subtask_descriptions:
            raise StoreValidation('subtask_plan_required')
        query_texts = [task_query]+([plan_text(subtask_descriptions)] if k_u else [])
        query_vectors = {}
        for text in query_texts:
            value = vectors.get(text)
            if value is None:
                value = self.get_cached(profile_id,text)
            if value is None:
                raise StoreValidation('missing_query_embedding')
            query_vectors[text] = _vector(value,dim)
        edge_rows = list(self.connection.execute('SELECT * FROM graph_hyperedges WHERE profile_id=?',(profile_id,)))
        if not edge_rows and self.connection.execute('SELECT 1 FROM graph_hyperedges LIMIT 1').fetchone():
            raise StoreValidation('no_index_for_embedding_profile')
        edges = {r['hyperedge_id']:json.loads(r['data_json']) for r in edge_rows}
        task_scores = []
        for row in edge_rows:
            cached = json.loads(self._require('embedding_cache','cache_id',row['cache_id'])['data_json'])
            value = _vector(cached['vector'],dim)
            task_scores.append({'hyperedge_id':row['hyperedge_id'],'score':_cosine(query_vectors[task_query],value)})
        task_hits = sorted(task_scores,key=lambda x:(-x['score'],x['hyperedge_id']))[:k_e]
        sub_hits = []
        if k_u:
            indexed = list(self.connection.execute(
                "SELECT DISTINCT n.node_id,n.data_json FROM graph_nodes n JOIN graph_memberships m USING(node_id) JOIN graph_hyperedges e USING(hyperedge_id) WHERE n.kind='subtask' AND e.profile_id=?",(profile_id,)))
            scored = []
            for row in indexed:
                data = json.loads(row['data_json']); value = self.get_cached(profile_id,data['text'])
                if value is None:
                    raise StoreValidation('missing_subtask_index_embedding')
                scored.append({'node_id':row['node_id'],'score':_cosine(query_vectors[plan_text(subtask_descriptions)],value)})
            sub_hits = sorted(scored,key=lambda x:(-x['score'],x['node_id']))[:k_u]
        sub_edges = set()
        for hit in sub_hits:
            attached = [r[0] for r in self.connection.execute(
                'SELECT hyperedge_id FROM graph_memberships WHERE node_id=?',(hit['node_id'],)) if r[0] in edges]
            hit['hyperedge_ids'] = sorted(attached);sub_edges.update(attached)
        union = sorted(sub_edges|{h['hyperedge_id'] for h in task_hits})
        versions = set()
        for edge in union:
            versions.update(r[0] for r in self.connection.execute(
                "SELECT n.version_id FROM graph_nodes n JOIN graph_memberships m USING(node_id) WHERE m.hyperedge_id=? AND n.kind='skill'",(edge,)))
        return {'status':'recalled' if edge_rows else 'empty_graph','embedding_profile_id':profile_id,
                'budget':{'k':k,'task_edges':k_e,'subtask_nodes':k_u},
                'task_route':task_hits,'subtask_route':sub_hits,'subtask_route_hyperedges':sorted(sub_edges),
                'union_hyperedge_ids':union,'candidate_versions':[self.get_skill_version(v) for v in sorted(versions)],
                'candidate_version_ids':sorted(versions),'ranking_policy':'none_id_order_only'}

    def graph_snapshot(self):
        def records(table):
            return [json.loads(r[0]) for r in self.connection.execute('SELECT data_json FROM '+table+' ORDER BY 1')]
        return {'schema_version':self.summary()['schema_version'],
                'nodes':records('graph_nodes'),'hyperedges':records('graph_hyperedges'),
                'memberships':[dict(r) for r in self.connection.execute('SELECT * FROM graph_memberships ORDER BY hyperedge_id,node_id')],
                'embedding_profiles':records('embedding_profiles'),
                'embedding_cache_count':self.connection.execute('SELECT COUNT(*) FROM embedding_cache').fetchone()[0],
                'jobs':[self.get_job(r[0]) for r in self.connection.execute('SELECT branch_id FROM learning_jobs ORDER BY branch_id')]}
