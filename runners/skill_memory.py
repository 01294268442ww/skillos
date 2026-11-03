#!/usr/bin/env python3
"""Extract from a stored main branch or return a two-route candidate union."""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default=str(ROOT/'storage/evidence.sqlite3'))
    parser.add_argument('--config',default='/home/ubuntu/.config/hyperskill/runtime.env')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-tokens',type=int,default=4096)
    parser.add_argument('--max-api-attempts',type=int,default=3)
    parser.add_argument('--llm-timeout',type=float,default=90)
    parser.add_argument('--embedding-timeout',type=float,default=60)
    commands = parser.add_subparsers(dest='command',required=True)
    learn = commands.add_parser('learn')
    learn.add_argument('--branch-id',required=True)
    learn.add_argument('--retry-failed',action='store_true')
    recall = commands.add_parser('recall')
    query = recall.add_mutually_exclusive_group(required=True)
    query.add_argument('--query'); query.add_argument('--query-file')
    recall.add_argument('--subtasks-file',help='Optional JSON list of already provided subtask descriptions.')
    recall.add_argument('--k',type=int,required=True,help='Explicit hyperedge/subtask recall budget; no skill top-k.')
    commands.add_parser('graph')
    job = commands.add_parser('job');job.add_argument('--branch-id',required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (directory/'operation.json').exists():
        print(json.dumps({'status':'error','error':{'category':'output_already_exists'}}))
        return 2
    logging.disable(logging.CRITICAL)
    model = embedding = store = None
    secrets = []
    operation = {'command':args.command,'status':'error'}
    try:
        from hyperskill.no_skill_runtime import load_runtime_config, AuditedOpenAIServerModel, redact
        from hyperskill.api_embedding import QwenEmbeddingClient
        from hyperskill.learning_store import LearningStore
        from hyperskill.learning import LearningService, safe_error, branch_trajectory
        config = load_runtime_config(args.config) if Path(args.config).is_file() else None
        if config is not None:
            secrets = [v for k,v in config.items() if v and any(
                marker in k.upper() for marker in ('API_KEY','TOKEN','SECRET','PASSWORD'))]
        store = LearningStore(args.db,secrets=secrets,create=False)
        if args.command == 'graph':
            operation = {'status':'snapshot','graph':store.graph_snapshot()}
        elif args.command == 'job':
            operation = {'status':'job','job':store.get_job(args.branch_id)}
        else:
            if args.command == 'learn':
                branch = store.get_branch(args.branch_id)
                if 'main' not in branch['roles']:
                    operation = {'status':'skipped_extra','branch_id':args.branch_id}
                elif not branch_trajectory(store,branch):
                    operation = {'status':'no_executable_trajectory','branch_id':args.branch_id}
                else:
                    operation = None
            if args.command == 'recall' or operation is None:
                if config is None:
                    raise RuntimeError('explicit_runtime_config_required')
                model = AuditedOpenAIServerModel(config,max_tokens=args.max_tokens,
                    max_api_attempts=args.max_api_attempts,request_timeout=args.llm_timeout)
                embedding = QwenEmbeddingClient(config,max_attempts=args.max_api_attempts,
                                                timeout=args.embedding_timeout)
                service = LearningService(store,model,embedding)
                if args.command == 'learn':
                    operation = service.learn(args.branch_id,retry_failed=args.retry_failed)
                else:
                    text = args.query if args.query is not None else Path(args.query_file).read_text(encoding='utf-8').strip()
                    descriptions = json.loads(Path(args.subtasks_file).read_text()) if args.subtasks_file else None
                    operation = service.recall(text,args.k,subtask_descriptions=descriptions)
    except Exception as error:
        operation = {'status':'error','error':safe_error(error) if 'safe_error' in locals()
                     else {'type':type(error).__name__}}
    finally:
        def clean(item):
            if isinstance(item,str):
                for secret in secrets:item=item.replace(secret,'[REDACTED]')
                return item
            if item is None or type(item) in (bool,int,float):return item
            if isinstance(item,list):return [clean(x) for x in item]
            if isinstance(item,dict):return {clean(str(k)):clean(v) for k,v in item.items()}
            return {'unserialized_type':type(item).__name__}
        artifacts = {'operation.json':operation,
            'llm_calls.json':model.call_ledger if model is not None else [],
            'embedding_calls.json':embedding.call_ledger if embedding is not None else [],
            'usage.json':{'llm':model.usage_summary() if model is not None else None,
                          'embedding':embedding.usage_summary() if embedding is not None else None}}
        if store is not None:
            artifacts['store_summary.json'] = store.summary()
            store.close()
        for name,payload in artifacts.items():
            with (directory/name).open('x',encoding='utf-8') as handle:
                json.dump(clean(payload),handle,ensure_ascii=False,indent=2);handle.write('\n')
        if model is not None:model.client.close()
        if embedding is not None:embedding.close()
    status = operation.get('status')
    ok = status in ('completed','already_completed','skipped_extra','no_executable_trajectory',
                    'recalled','empty_graph','snapshot','job')
    print(json.dumps({'status':status,'output_directory':str(directory.resolve()),
                      'new_llm_calls':len(artifacts['llm_calls.json']),
                      'new_embedding_calls':len(artifacts['embedding_calls.json']),
                      'error':operation.get('error')},ensure_ascii=False))
    return 0 if ok else 2


if __name__=='__main__':raise SystemExit(main())
