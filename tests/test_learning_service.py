"""Learning-service integration with real temporary SQLite and fake API clients.

All model responses, observations and vectors below are explicit offline fixtures.
No SDK request is sent. Server's existing numpy is used for encode's real interface.
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

import numpy as np

REPO = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
PACKAGE = 'tested_learning_service_package'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(REPO/'hyperskill')]
sys.modules[PACKAGE] = package


def load(name):
    spec=importlib.util.spec_from_file_location(PACKAGE+'.'+name,REPO/'hyperskill'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


core=load('evidence_store')
graph=load('learning_store')
service=load('learning')
PROFILE={'model':'offline-qwen-fixture','endpoint':'https://example.invalid/v1','dimensions':3}


def extraction():
    return {'subtasks':[{'name':'Inspect source','description':'Inspect the authoritative fixture source'}],
            'skills':[{'name':'Evidence checked procedure',
                       'content':'Inspect the authoritative source and verify its scope before extracting facts.',
                       'conditions':'When a fixture task names an authoritative source.',
                       'goal_refs':['Inspect the authoritative fixture source'],'evidence_steps':[1]}],
            'fragment':'A faithful offline fixture procedure.'}


class FixtureApiError(RuntimeError):
    pass


class FakeModel:
    def __init__(self,responses):
        self.responses=list(responses)
        self.calls=[];self.call_ledger=[];self.phase='fixture'
        self.public_configuration={'model':'offline-deepseek-fixture','endpoint':'https://example.invalid/v1'}

    def __call__(self,messages,**kwargs):
        self.calls.append(copy.deepcopy(messages))
        value=self.responses.pop(0)
        row={'sequence':len(self.call_ledger)+1,'phase':self.phase,
             'usage':None if isinstance(value,Exception) else {'prompt_tokens':9,'completion_tokens':2,'total_tokens':11},
             'outcome':'request_failed' if isinstance(value,Exception) else 'response_received'}
        self.call_ledger.append(row)
        if isinstance(value,Exception):raise value
        return types.SimpleNamespace(content=value)


class FakeEmbedding:
    def __init__(self,failures=0):
        self.profile=copy.deepcopy(PROFILE)
        self.public_configuration=copy.deepcopy(PROFILE)
        self.call_ledger=[];self.calls=[];self.failures=failures

    def encode(self,texts,convert_to_numpy,show_progress_bar):
        self.calls.append(list(texts))
        if self.failures:
            self.failures-=1
            self.call_ledger.append({'sequence':len(self.call_ledger)+1,'usage':None,'outcome':'request_failed'})
            raise FixtureApiError('controlled embedding fixture failure')
        self.call_ledger.append({'sequence':len(self.call_ledger)+1,'usage':{'total_tokens':7},'outcome':'response_received'})
        return np.asarray([[1.0,0.0,0.0] for _ in texts],dtype=np.float64)


class LearningServiceBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='hyperskill-learning-service-test-')
        self.path=Path(self.tmp.name)/'evidence.sqlite3'
        self.store=graph.LearningStore(self.path)
        self.protocol_id=self.store.register_protocol({'model':'fixture-only','memory_provider':None})
        self.trajectory=[{'name':'action','tool_calls':[{'name':'crawl_page',
            'arguments':{'url':'https://example.invalid/source','query':'Read the fixture'}}],
            'obs':'Recorded fixture observation; no HTTP request occurred.'}]

    def tearDown(self):
        with suppress(Exception):self.store.close()
        self.tmp.cleanup()

    def branch(self,roles=None,trajectory=None,task_id='fixture-task'):
        branch_id=task_id+':main'
        self.store.register_task(task_id,'Inspect a fixture source.',main_branch_id=branch_id)
        self.store.register_branch(branch_id,task_id,self.protocol_id,roles=roles or ['main'])
        payload=self.trajectory if trajectory is None else trajectory
        evidence=self.store.put_evidence('trajectory',payload)
        self.store.attach_branch_evidence(branch_id,'trajectory.json',evidence)
        # Engineering completion and checker-looking data are deliberately not a verdict.
        self.store.record_branch_result(branch_id,{'status':'passed','is_correct':True,
            'checker_verdict':'not an actual checker','final_answer':'Fixture output only.'})
        return branch_id

    def test_prompt_uses_real_recorded_fields_and_does_not_infer_checker_outcome(self):
        branch_id=self.branch()
        model=FakeModel([json.dumps(extraction())]);embedding=FakeEmbedding()
        learned=service.LearningService(self.store,model,embedding).learn(branch_id)
        self.assertEqual(learned['status'],'completed')
        prompt=model.calls[0][0]['content'][0]['text']
        source=json.loads(prompt.split('\nExecution evidence:\n',1)[1])
        self.assertEqual(source['trajectory'],self.trajectory)
        self.assertIn('tool_calls',source['trajectory'][0]);self.assertIn('obs',source['trajectory'][0])
        self.assertEqual(source['provided_version_ids'],[])
        self.assertNotIn('is_correct',source);self.assertNotIn('status',source)
        self.assertNotIn('checker_verdict',source)
        self.assertIn('Do not assign correctness',prompt)
        self.assertNotIn('SUCCESSFUL',prompt);self.assertNotIn('INCORRECT',prompt)
        self.assertIsNone(self.store.get_skill_version(learned['version_ids'][0])['value'])

    def test_main_reused_as_base_learns_once_with_zero_repeat_api_calls(self):
        branch_id=self.branch(roles=['main','base'])
        model=FakeModel([json.dumps(extraction())]);embedding=FakeEmbedding()
        learner=service.LearningService(self.store,model,embedding)
        first=learner.learn(branch_id);before=self.store.summary()
        second=learner.learn(branch_id)
        self.assertEqual(first['status'],'completed');self.assertEqual(second['status'],'already_completed')
        self.assertEqual((len(model.calls),len(embedding.calls)),(1,1))
        self.assertEqual(second['new_llm_calls'],0);self.assertEqual(second['new_embedding_calls'],0)
        self.assertEqual(self.store.summary(),before)

    def constructor_spies(self):
        model_constructor=Mock(side_effect=AssertionError('API model must not be constructed'))
        embedding_constructor=Mock(side_effect=AssertionError('Embedding client must not be constructed'))
        model_module=types.ModuleType(PACKAGE+'.no_skill_runtime')
        model_module.AuditedOpenAIServerModel=model_constructor
        # Eligibility fixtures return before this serializer is used.
        model_module.redact=lambda value,secrets:value
        embedding_module=types.ModuleType(PACKAGE+'.api_embedding')
        embedding_module.QwenEmbeddingClient=embedding_constructor
        return {model_module.__name__:model_module,embedding_module.__name__:embedding_module},model_constructor,embedding_constructor

    def test_extra_gate_precedes_source_checks_and_both_client_constructors(self):
        self.branch()
        self.store.register_branch('fixture-task:extra','fixture-task',self.protocol_id,roles=['base','candidate'])
        before=self.store.summary()
        modules,model,embedding=self.constructor_spies()
        with patch.dict(sys.modules,modules):
            receipt=service.process_main_from_config(self.path,'fixture-task:extra',{})
        self.assertEqual(receipt['status'],'skipped_extra')
        model.assert_not_called();embedding.assert_not_called()
        self.assertEqual(self.store.summary(),before)

    def test_empty_native_trajectory_does_not_construct_either_client(self):
        branch_id=self.branch(trajectory=[])
        modules,model,embedding=self.constructor_spies()
        with patch.dict(sys.modules,modules):
            receipt=service.process_main_from_config(self.path,branch_id,{})
        self.assertEqual(receipt['status'],'no_executable_trajectory')
        model.assert_not_called();embedding.assert_not_called()
        self.assertEqual(self.store.summary()['counts']['learning_jobs'],0)

    def test_embedding_failure_retains_checkpoint_and_retry_does_not_repeat_llm(self):
        branch_id=self.branch();expected=extraction()
        model=FakeModel([json.dumps(expected)]);embedding=FakeEmbedding(failures=1)
        learner=service.LearningService(self.store,model,embedding)
        with self.assertRaises(FixtureApiError):learner.learn(branch_id)
        failed=self.store.get_job(branch_id)
        self.assertEqual(failed['state'],'failed');self.assertEqual(failed['checkpoint'],expected)
        self.assertEqual((len(model.calls),len(embedding.calls)),(1,1))
        first_embedding=self.store.get_evidence(failed['embedding_ledger_refs'][0])['payload']
        self.assertIsNone(first_embedding['calls'][0]['usage'])
        self.assertEqual(learner.learn(branch_id)['status'],'failed')
        completed=learner.learn(branch_id,retry_failed=True)
        self.assertEqual(completed['status'],'completed')
        self.assertEqual(completed['new_llm_calls'],0);self.assertEqual(completed['new_embedding_calls'],1)
        self.assertEqual((len(model.calls),len(embedding.calls)),(1,2))
        self.assertEqual(len(self.store.get_job(branch_id)['llm_ledger_refs']),1)

    def test_native_db_handler_failure_keeps_paid_maintenance_ledger_and_closes_clients(self):
        branch_id=self.branch()
        model=FakeModel([]);embedding=FakeEmbedding()
        llm_call={'sequence':1,'phase':'routine_skill_extraction',
                  'usage':{'prompt_tokens':9,'completion_tokens':2,'total_tokens':11}}
        embedding_call={'sequence':1,'usage':{'total_tokens':7}}
        llm_summary={'request_attempts':1,'usage_complete':True,
                     'total_usage':{'prompt_tokens':9,'completion_tokens':2,'total_tokens':11}}
        embedding_summary={'request_attempts':1,'usage_complete':True,'total_tokens':7}
        model.client=types.SimpleNamespace(close=Mock())
        model.usage_summary=Mock(return_value=llm_summary)
        embedding.close=Mock()
        embedding.usage_summary=Mock(return_value=embedding_summary)
        model_module=types.ModuleType(PACKAGE+'.no_skill_runtime')
        model_module.AuditedOpenAIServerModel=Mock(return_value=model)
        model_module.redact=lambda value,secrets:value
        embedding_module=types.ModuleType(PACKAGE+'.api_embedding')
        embedding_module.QwenEmbeddingClient=Mock(return_value=embedding)
        def fail_after_paid_calls(learner,branch,retry_failed=False):
            self.assertEqual(branch,branch_id)
            learner.model.call_ledger.append(copy.deepcopy(llm_call))
            learner.embedding.call_ledger.append(copy.deepcopy(embedding_call))
            # The handler cannot write its ledger back to this real temporary DB.
            learner.store.close()
            raise core.StoreError('controlled_db_failure_handler_failed')
        with patch.dict(sys.modules,{model_module.__name__:model_module,
                                     embedding_module.__name__:embedding_module}):
            with patch.object(service.LearningService,'learn',autospec=True,
                              side_effect=fail_after_paid_calls):
                receipt=service.process_main_from_config(self.path,branch_id,{})
        self.assertEqual(receipt['status'],'error')
        self.assertEqual(receipt['error']['category'],'controlled_db_failure_handler_failed')
        self.assertEqual(receipt['maintenance_api']['llm_calls'],[llm_call])
        self.assertEqual(receipt['maintenance_api']['embedding_calls'],[embedding_call])
        self.assertEqual(receipt['maintenance_api']['usage'],{'llm':llm_summary,'embedding':embedding_summary})
        model.client.close.assert_called_once_with();embedding.close.assert_called_once_with()
        model_module.AuditedOpenAIServerModel.assert_called_once()
        embedding_module.QwenEmbeddingClient.assert_called_once()
        self.assertEqual(self.store.summary()['counts']['branches'],1)

    def test_invalid_llm_schema_retains_failed_job_and_billed_usage(self):
        branch_id=self.branch();invalid=extraction();invalid['skills'][0]['goal_refs']=['invented objective']
        model=FakeModel([json.dumps(invalid)]);embedding=FakeEmbedding()
        with self.assertRaises(core.StoreValidation):service.LearningService(self.store,model,embedding).learn(branch_id)
        job=self.store.get_job(branch_id)
        self.assertEqual(job['state'],'failed');self.assertIsNone(job['checkpoint'])
        self.assertEqual(len(model.calls),1);self.assertEqual(len(embedding.calls),0)
        ledger=self.store.get_evidence(job['llm_ledger_refs'][0])['payload']['calls']
        self.assertEqual(ledger[0]['usage']['total_tokens'],11)
        self.assertEqual(self.store.summary()['counts']['skill_versions'],0)

    def test_recall_decomposes_then_embeds_task_and_aggregated_plan_without_value_filter(self):
        branch_id=self.branch()
        plan={'subtasks':[{'name':'Inspect','description':'Check authoritative source'},
                          {'name':'Extract','description':'Read exact stated constraints'}]}
        model=FakeModel([json.dumps(extraction()),json.dumps(plan)]);embedding=FakeEmbedding()
        learner=service.LearningService(self.store,model,embedding)
        learned=learner.learn(branch_id)
        recalled=learner.recall('New fixture retrieval task',k=4)
        self.assertEqual(recalled['status'],'recalled')
        self.assertEqual(recalled['plan_source'],'deepseek_query_decomposition')
        self.assertEqual(recalled['subtask_descriptions'],[x['description'] for x in plan['subtasks']])
        self.assertEqual(set(embedding.calls[-1]),{
            'New fixture retrieval task','Check authoritative source Read exact stated constraints'})
        self.assertEqual(model.call_ledger[-1]['phase'],'recall_query_decomposition')
        self.assertEqual(recalled['candidate_version_ids'],learned['version_ids'])
        self.assertTrue(all(v['value'] is None for v in recalled['candidate_versions']))
        evidence=self.store.get_evidence(recalled['evidence_id'])['payload']
        self.assertEqual(len(evidence['llm_calls']),1);self.assertEqual(len(evidence['embedding_calls']),1)
        self.assertEqual(evidence['result']['ranking_policy'],'none_id_order_only')

    def test_api_failures_are_recorded_and_raised_instead_of_returning_empty_graph(self):
        for label,model,embedding,k in (
            ('decomposition',FakeModel([FixtureApiError('controlled model fixture failure')]),FakeEmbedding(),2),
            ('embedding',FakeModel([]),FakeEmbedding(failures=1),1)):
            with self.subTest(label=label):
                before=self.store.summary()['counts']['evidence']
                with self.assertRaises(FixtureApiError):service.LearningService(self.store,model,embedding).recall('Failed fixture '+label,k=k)
                self.assertEqual(self.store.summary()['counts']['evidence'],before+1)
        failures=[json.loads(row[0]) for row in self.store.connection.execute('SELECT data_json FROM evidence')]
        self.assertTrue(all(row['kind']=='candidate_recall_failure' for row in failures))

    def test_duplicate_json_keys_and_unanchored_steps_or_goals_are_rejected(self):
        with self.assertRaises(core.StoreValidation):service.parse_object('{"subtasks":[],"subtasks":[]}')
        for steps,goals in (([0],None),([2],None),([True],None),([1],['not a declared objective'])):
            with self.subTest(steps=steps,goals=goals):
                value=extraction();value['skills'][0]['evidence_steps']=steps
                if goals is not None:value['skills'][0]['goal_refs']=goals
                with self.assertRaises(core.StoreValidation):service.validate_extraction(value,step_count=1)
        value=extraction();value['skills'][0]['value']=1
        with self.assertRaises(core.StoreValidation):service.validate_extraction(value,step_count=1)


if __name__=='__main__':
    unittest.main(verbosity=2)
