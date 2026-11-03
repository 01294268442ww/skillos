"""SQLite learning/graph behavior checks; all observations here are fixtures.

No server, LLM or embedding API is contacted. HYPERSKILL_REPO selects the source.
"""
from contextlib import suppress
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
# Avoid the application package's unrelated eager imports.
PACKAGE = 'tested_learning_store_package'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(REPO/'hyperskill')]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE+'.'+name, REPO/'hyperskill'/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load('evidence_store')
learning = load('learning_store')
LearningStore = learning.LearningStore
StoreValidation = core.StoreValidation
StoreConflict = core.StoreConflict
StoreNotFound = core.StoreNotFound
PROFILE = {'model':'offline-qwen-fixture','endpoint':'https://example.invalid/v1','dimensions':3}
EXTRACTOR = {'model':'offline-deepseek-fixture','prompt_version':'fixture-v1'}


class LearningStoreBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='hyperskill-learning-store-test-')
        self.path = Path(self.tmp.name)/'evidence.sqlite3'
        self.opened = []
        self.store = self.open_store()
        self.protocol_id = self.store.register_protocol({'model':'fixture-only','memory_provider':None})

    def tearDown(self):
        for store in reversed(self.opened):
            with suppress(Exception):store.close()
        self.tmp.cleanup()

    def open_store(self, path=None, **kwargs):
        store = LearningStore(path or self.path, **kwargs)
        self.opened.append(store)
        return store

    def branch(self, task_id='q1', roles=None, with_result=True, with_trajectory=True):
        branch_id = task_id+':main'
        self.store.register_task(task_id, 'Fixture task '+task_id, main_branch_id=branch_id)
        self.store.register_branch(branch_id,task_id,self.protocol_id,roles=roles or ['main'])
        if with_trajectory:
            trajectory = [{'name':'action','tool_calls':[{'name':'crawl_page','arguments':{'url':'https://example.invalid/source'}}],
                           'obs':'A fixture observation, not an actual HTTP request.'}]
            evidence_id = self.store.put_evidence('trajectory',trajectory)
            self.store.attach_branch_evidence(branch_id,'trajectory.json',evidence_id)
        if with_result:
            self.store.record_branch_result(branch_id,{'status':'fixture_only','final_answer':'Fixture observation.'})
        return branch_id

    def extraction(self, label='A', skills=1):
        description = 'Inspect fixture goal '+label
        return {'subtasks':[{'name':'Inspect '+label,'description':description}],
                'skills':[{'name':'Fixture procedure '+label+str(i),
                           'content':'Apply fixture procedure '+label+str(i)+'. Verify the fixture evidence.',
                           'conditions':'Only the explicit offline '+label+' scenario.',
                           'goal_refs':[description],'evidence_steps':[1]} for i in range(skills)],
                'fragment':'Fixture reusable lesson '+label}

    def vectors(self, branch_id, extraction, sub=(0,1,0), edge=(1,0,0)):
        task = self.store.get_task(self.store.get_branch(branch_id)['task_id'])['task']
        values = {learning.subtask_text(st):list(sub) for st in extraction['subtasks']}
        values[learning.episode_text(task,extraction['fragment'])] = list(edge)
        return values

    def claimed(self, branch_id):
        claim = self.store.claim(branch_id,EXTRACTOR,PROFILE)
        self.assertEqual(claim['status'],'claimed')
        return claim

    def completed(self, task_id='q1', label='A', skills=1, sub=(0,1,0), edge=(1,0,0), roles=None):
        branch_id = self.branch(task_id,roles=roles)
        extraction = self.extraction(label,skills)
        claim = self.claimed(branch_id)
        self.store.checkpoint(branch_id,claim['owner_token'],extraction,[])
        receipt = self.store.commit_extraction(branch_id,claim['owner_token'],extraction,
            self.vectors(branch_id,extraction,sub,edge),PROFILE,[])
        return branch_id,extraction,receipt

    def test_v1_migration_preserves_existing_rows_and_opaque_frozen_value(self):
        legacy = Path(self.tmp.name)/'legacy.sqlite3'
        conn = sqlite3.connect(legacy)
        conn.executescript(core.V1_SCHEMA)
        conn.execute('PRAGMA application_id='+str(core.APPLICATION_ID))
        conn.execute('PRAGMA user_version=1')
        skill_data = core.canonical_json({'skill_id':'legacy-skill'})
        conn.execute('INSERT INTO skills VALUES(?,?,?)',('legacy-skill',skill_data,'old-time'))
        version_data = {'version_id':'legacy-version','skill_id':'legacy-skill','version':1,
                        'content':'Unchanged legacy fixture','conditions':'Legacy','goal_refs':['old-goal'],
                        'origin_kind':'legacy_fixture','parent_versions':[],'content_sha256':'old-digest'}
        conn.execute('INSERT INTO skill_versions VALUES(?,?,?,?,?,?,?)',
            ('legacy-version','legacy-skill',1,'old-digest','[0.2,17]',core.canonical_json(version_data),'old-time'))
        before = conn.execute('SELECT * FROM skill_versions').fetchall();conn.commit();conn.close()
        upgraded = self.open_store(legacy,create=False)
        self.assertEqual(upgraded.summary()['schema_version'],core.SCHEMA_VERSION)
        self.assertEqual(upgraded.connection.execute('SELECT * FROM skill_versions').fetchall()[0][:],before[0])
        self.assertEqual(upgraded.get_skill_version('legacy-version')['value'],[0.2,17])
        self.assertEqual(upgraded.summary()['counts']['learning_jobs'],0)
        self.assertEqual(upgraded.integrity_check(),{'integrity':['ok'],'foreign_key_violations':[]})

    def test_main_and_reused_base_share_one_claim(self):
        branch_id = self.branch(roles=['main','base'])
        first = self.claimed(branch_id)
        before = self.store.summary()
        second = self.store.claim(branch_id,EXTRACTOR,PROFILE)
        self.assertEqual(second['status'],'busy')
        self.assertEqual(first['job_id'],second['job_id'])
        self.assertEqual(self.store.summary(),before)
        self.assertEqual(before['counts']['learning_jobs'],1)

    def test_extra_branch_is_skipped_before_profiles_or_jobs(self):
        self.branch()
        self.store.register_branch('q1:extra','q1',self.protocol_id,roles=['base','candidate'])
        before = self.store.summary()
        result = self.store.claim('q1:extra',object(),object())
        self.assertEqual(result['status'],'skipped_extra')
        self.assertEqual(self.store.summary(),before)
        self.assertEqual(before['counts']['embedding_profiles'],0)

    def test_unfinished_main_or_missing_source_cannot_claim(self):
        for task_id,kwargs in [('unfinished',{'with_result':False}),('missing-source',{'with_trajectory':False})]:
            branch_id = self.branch(task_id,**kwargs)
            with self.subTest(task_id=task_id):
                with self.assertRaises(StoreValidation):self.store.claim(branch_id,EXTRACTOR,PROFILE)
        self.assertEqual(self.store.summary()['counts']['learning_jobs'],0)

    def test_checkpoint_resume_requires_explicit_retry_and_preserves_unknown_usage(self):
        branch_id = self.branch();extraction = self.extraction();claim = self.claimed(branch_id)
        ledger = [{'sequence':1,'usage':None,'outcome':'request_failed'}]
        self.store.checkpoint(branch_id,claim['owner_token'],extraction,ledger)
        self.store.fail(branch_id,claim['owner_token'],{'type':'FixtureNetworkError'},embedding_ledger=ledger)
        failed = self.store.claim(branch_id,EXTRACTOR,PROFILE)
        self.assertEqual(failed['status'],'failed')
        self.assertEqual(failed['checkpoint'],extraction)
        retry = self.store.claim(branch_id,EXTRACTOR,PROFILE,retry_failed=True)
        self.assertNotEqual(retry['owner_token'],claim['owner_token'])
        self.assertEqual(retry['checkpoint'],extraction)
        with self.assertRaises(StoreConflict):self.store.fail(branch_id,claim['owner_token'],{'type':'StaleFixture'})
        job = self.store.get_job(branch_id)
        payload = self.store.get_evidence(job['llm_ledger_refs'][0])['payload']
        self.assertIsNone(payload['calls'][0]['usage'])
        self.store.commit_extraction(branch_id,retry['owner_token'],extraction,
                                    self.vectors(branch_id,extraction),PROFILE,[])
        self.assertEqual(self.store.get_job(branch_id)['state'],'completed')

    def test_commit_requires_matching_checkpoint(self):
        branch_id = self.branch();extraction=self.extraction();claim=self.claimed(branch_id)
        before=self.store.summary()
        with self.assertRaises(StoreValidation):self.store.commit_extraction(
            branch_id,claim['owner_token'],extraction,self.vectors(branch_id,extraction),PROFILE,[])
        self.assertEqual(self.store.summary(),before)

    def test_bad_vector_commit_rolls_back_every_graph_or_version_write(self):
        branch_id=self.branch();extraction=self.extraction();claim=self.claimed(branch_id)
        self.store.checkpoint(branch_id,claim['owner_token'],extraction,[])
        vectors=self.vectors(branch_id,extraction)
        vectors[sorted(vectors)[-1]]=[1.0,2.0]
        before=self.store.summary()
        with self.assertRaises(StoreValidation):self.store.commit_extraction(
            branch_id,claim['owner_token'],extraction,vectors,PROFILE,[])
        self.assertEqual(self.store.summary(),before)
        self.assertEqual(self.store.get_job(branch_id)['state'],'running')

    def test_commit_can_use_same_profile_cache_without_new_api_rows(self):
        branch_id=self.branch();extraction=self.extraction();claim=self.claimed(branch_id)
        self.store.checkpoint(branch_id,claim['owner_token'],extraction,[])
        profile_id=self.store.register_embedding_profile(PROFILE)
        for text,vector in self.vectors(branch_id,extraction).items():self.store.put_cached(profile_id,text,vector)
        receipt=self.store.commit_extraction(branch_id,claim['owner_token'],extraction,{},PROFILE,[])
        self.assertEqual(receipt['status'],'completed')
        self.assertEqual(self.store.summary()['counts']['learning_api_events'],0)

    def test_completed_branch_never_gets_a_second_extraction(self):
        branch_id,_,receipt=self.completed()
        before=self.store.summary()
        second=self.store.claim(branch_id,{'model':'a changed model'},PROFILE,retry_failed=True)
        self.assertEqual(second['status'],'already_completed')
        self.assertEqual(second['receipt'],receipt)
        self.assertEqual(self.store.summary(),before)

    def test_cache_is_profile_scoped_immutable_and_dimension_checked(self):
        first=self.store.register_embedding_profile(PROFILE)
        second=self.store.register_embedding_profile({**PROFILE,'model':'a different fixture model'})
        self.store.put_cached(first,'same public fixture',[1,0,0])
        self.assertEqual(self.store.get_cached(first,'same public fixture'),[1.0,0.0,0.0])
        self.assertIsNone(self.store.get_cached(second,'same public fixture'))
        with self.assertRaises(StoreConflict):self.store.put_cached(first,'same public fixture',[0,1,0])
        for vector in ([1,0],[0,0,0],[float('nan'),0,1],[True,0,1]):
            with self.subTest(vector=vector):
                with self.assertRaises(StoreValidation):self.store.put_cached(first,'invalid vector',vector)

    def test_wrong_profile_commit_does_not_modify_checkpoint_or_graph(self):
        branch_id=self.branch();extraction=self.extraction();claim=self.claimed(branch_id)
        self.store.checkpoint(branch_id,claim['owner_token'],extraction,[])
        before=self.store.summary()
        with self.assertRaises(StoreConflict):self.store.commit_extraction(
            branch_id,claim['owner_token'],extraction,self.vectors(branch_id,extraction),
            {**PROFILE,'model':'wrong model'},[])
        self.assertEqual(self.store.summary(),before)

    def test_versions_keep_source_evidence_and_unvalued_state(self):
        branch_id,_,receipt=self.completed()
        version=self.store.get_skill_version(receipt['version_ids'][0])
        self.assertIsNone(version['value'])
        self.assertEqual({x['branch_id'] for x in version['evidence_refs']},{branch_id})
        self.assertEqual({x['relation'] for x in version['evidence_refs']},{'source'})
        self.assertEqual(len(version['evidence_refs']),2)
        self.assertEqual(self.store.get_branch(branch_id)['supplied_versions'],[])

    def test_identical_content_reuses_version_and_appends_distinct_sources(self):
        _,_,first=self.completed('first','A')
        _,_,second=self.completed('second','A')
        self.assertEqual(first['version_ids'],second['version_ids'])
        self.assertEqual(self.store.summary()['counts']['skill_versions'],1)
        version=self.store.get_skill_version(first['version_ids'][0])
        self.assertEqual(len({x['branch_id'] for x in version['evidence_refs']}),2)
        self.assertIsNone(version['value'])

    def test_changed_conditions_never_merge_by_name_or_semantic_similarity(self):
        _,_,first=self.completed('first','A')
        branch_id=self.branch('second');extraction=self.extraction('A')
        extraction['skills'][0]['conditions']='A materially different explicit condition.'
        claim=self.claimed(branch_id);self.store.checkpoint(branch_id,claim['owner_token'],extraction,[])
        second=self.store.commit_extraction(branch_id,claim['owner_token'],extraction,
                                          self.vectors(branch_id,extraction),PROFILE,[])
        self.assertNotEqual(first['version_ids'],second['version_ids'])
        self.assertEqual(self.store.summary()['counts']['skill_versions'],2)

    def test_graph_and_jobs_survive_close_and_reopen(self):
        self.completed();expected=self.store.graph_snapshot();self.store.close()
        reopened=self.open_store(create=False)
        self.assertEqual(reopened.graph_snapshot(),expected)
        self.assertEqual(reopened.integrity_check(),{'integrity':['ok'],'foreign_key_violations':[]})

    def test_dual_routes_are_subtask_and_episode_then_full_union(self):
        _,_,b=self.completed('qB','B',sub=(0,0,1),edge=(1,0,0))
        _,_,c=self.completed('qC','C',sub=(0,0.8,0.6),edge=(0.8,0.6,0))
        _,_,a=self.completed('qA','A',sub=(0,1,0),edge=(0,0,1))
        result=self.store.recall('new task query',['new plan'],
            {'new task query':[1,0,0],'new plan':[0,1,0]},PROFILE,k=4)
        self.assertEqual({x['hyperedge_id'] for x in result['task_route']},{b['hyperedge_id'],c['hyperedge_id']})
        self.assertEqual(set(result['subtask_route_hyperedges']),{a['hyperedge_id'],c['hyperedge_id']})
        self.assertEqual(set(result['union_hyperedge_ids']),{a['hyperedge_id'],b['hyperedge_id'],c['hyperedge_id']})
        self.assertEqual(set(result['candidate_version_ids']),set(a['version_ids']+b['version_ids']+c['version_ids']))
        self.assertEqual(result['ranking_policy'],'none_id_order_only')
        self.assertTrue(all(x['value'] is None for x in result['candidate_versions']))

    def test_skill_union_is_not_truncated_to_recall_budget(self):
        _,_,receipt=self.completed(skills=6)
        result=self.store.recall('task query',['unused plan'],{'task query':[1,0,0]},PROFILE,k=1)
        self.assertEqual(result['candidate_version_ids'],receipt['version_ids'])
        self.assertEqual(len(result['candidate_versions']),6)
        self.assertEqual(result['subtask_route'],[])
        self.assertEqual(result['budget'],{'k':1,'task_edges':1,'subtask_nodes':0})

    def test_wrong_query_dimensions_or_profile_are_not_empty_recall(self):
        self.completed()
        for vectors,profile in [({'q':[1,0]},PROFILE),({'q':[1,0,0]},{**PROFILE,'model':'wrong profile'})]:
            with self.subTest(profile=profile):
                with self.assertRaises(StoreValidation):self.store.recall('q',[],vectors,profile,k=1)

    def test_empty_graph_is_distinct_from_invalid_vector(self):
        result=self.store.recall('q',[],{'q':[1,0,0]},PROFILE,k=1)
        self.assertEqual(result['status'],'empty_graph')
        self.assertEqual(result['candidate_versions'],[])
        with self.assertRaises(StoreValidation):self.store.recall('q',[],{'q':[0,0,0]},PROFILE,k=1)

    def test_embedding_total_only_usage_is_preserved_without_chat_fields(self):
        branch_id=self.branch();extraction=self.extraction();claim=self.claimed(branch_id)
        self.store.checkpoint(branch_id,claim['owner_token'],extraction,[])
        ledger=[{'sequence':1,'usage':{'total_tokens':11}}]
        receipt=self.store.commit_extraction(branch_id,claim['owner_token'],extraction,
            self.vectors(branch_id,extraction),PROFILE,ledger)
        payload=self.store.get_evidence(receipt['embedding_ledger_refs'][0])['payload']
        self.assertEqual(payload['calls'],ledger)
        self.assertNotIn('prompt_tokens',payload['calls'][0]['usage'])


if __name__=='__main__':
    unittest.main(verbosity=2)
