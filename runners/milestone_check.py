#!/usr/bin/env python3
"""Freeze task milestones or check a fixed group of recorded branches."""
import argparse
import json
import logging
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def safe_error(error):
    output = {'type': type(error).__name__}
    category = getattr(error, 'category', None)
    if isinstance(category, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,79}', category):
        output['category'] = category
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=str(ROOT/'storage/evidence.sqlite3'))
    parser.add_argument('--config', default='/home/ubuntu/.config/hyperskill/runtime.env')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--max-api-attempts', type=int, choices=(1,), default=1,
                        help='One request per check round; SDK retries cannot expand the round budget.')
    parser.add_argument('--request-timeout', type=float, default=90)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--task-id', required=True)
    prepare.add_argument('--retry-failed', action='store_true')
    check = commands.add_parser('check')
    check.add_argument('--branch-id', action='append', required=True,
                       help='Repeat to share the same three-round budget across a frozen branch group.')
    show_spec = commands.add_parser('show-spec')
    show_spec.add_argument('--task-id', required=True)
    show_check = commands.add_parser('show-check')
    show_check.add_argument('--session-id', required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    if directory.exists():
        print(json.dumps({'status': 'error', 'error': {'category': 'output_already_exists'}}))
        return 2
    directory.mkdir(parents=True, mode=0o700)
    logging.disable(logging.CRITICAL)
    operation = {'status': 'error', 'command': args.command}
    maintenance = {'llm_calls': [], 'usage': {}}
    secrets = []
    summary = None
    try:
        from hyperskill.no_skill_runtime import load_runtime_config
        from hyperskill.milestone_store import MilestoneStore
        config = load_runtime_config(args.config) if Path(args.config).is_file() else None
        if config is not None:
            secrets = [value for key, value in config.items() if value and any(
                marker in key.upper() for marker in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))]
        if args.command in ('show-spec', 'show-check'):
            with MilestoneStore(args.db, secrets=secrets, create=False) as store:
                if args.command == 'show-spec':
                    operation = {'status': 'spec', 'specification': store.get_spec(args.task_id)}
                else:
                    operation = {'status': 'check', 'check': store.get_check_session(args.session_id)}
        else:
            if config is None:
                from hyperskill.evidence_store import StoreValidation
                raise StoreValidation('explicit_runtime_config_required')
            from hyperskill.milestones import prepare_task_from_config, check_branches_from_config
            limits = {'max_tokens': args.max_tokens, 'max_api_attempts': args.max_api_attempts,
                      'request_timeout': args.request_timeout}
            if args.command == 'prepare':
                operation = prepare_task_from_config(args.db, args.task_id, config,
                    retry_failed=args.retry_failed, **limits)
            else:
                operation = check_branches_from_config(args.db, args.branch_id, config, **limits)
            maintenance = operation.pop('maintenance_api', maintenance)
        with MilestoneStore(args.db, secrets=secrets, create=False) as store:
            summary = store.summary()
    except Exception as error:
        operation = {'status': 'error', 'command': args.command, 'error': safe_error(error)}
    finally:
        def clean(value):
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, '[REDACTED]')
                return value
            if value is None or type(value) in (bool, int, float):
                return value
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {clean(str(key)): clean(item) for key, item in value.items()}
            return {'unserialized_type': type(value).__name__}
        artifacts = {'operation.json': operation, 'llm_calls.json': maintenance['llm_calls'],
                     'usage.json': maintenance['usage']}
        if summary is not None:
            artifacts['store_summary.json'] = summary
        for name, payload in artifacts.items():
            with (directory/name).open('x', encoding='utf-8') as handle:
                json.dump(clean(payload), handle, ensure_ascii=False, indent=2)
                handle.write('\n')
    status = operation.get('status')
    print(json.dumps({'status': status, 'output_directory': str(directory.resolve()),
                      'session_id': operation.get('session_id'), 'spec_id': operation.get('spec_id'),
                      'new_llm_calls': len(maintenance['llm_calls']), 'error': operation.get('error')},
                     ensure_ascii=False))
    return 0 if status in ('completed', 'already_completed', 'unknown', 'spec', 'check') else 2


if __name__ == '__main__':
    raise SystemExit(main())
