#!/usr/bin/env python3
"""Initialize, import, inspect and explicitly add skill versions to SQLite."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT/"storage/evidence.sqlite3"))
    parser.add_argument("--config", default="/home/ubuntu/.config/hyperskill/runtime.env",
                        help="Existing private runtime config, used only to reject actual secrets.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("summary")
    commands.add_parser("check")
    cmd = commands.add_parser("import-run")
    cmd.add_argument("--run-dir", required=True)
    cmd = commands.add_parser("show-task")
    cmd.add_argument("--task-id", required=True)
    cmd = commands.add_parser("show-version")
    cmd.add_argument("--version-id", required=True)
    cmd = commands.add_parser("list-versions")
    cmd.add_argument("--skill-id", required=True)
    cmd = commands.add_parser("put-version")
    cmd.add_argument("--file", required=True, help="Explicit JSON skill/version payload; never extracted automatically.")
    args = parser.parse_args()
    try:
        from hyperskill.evidence_store import EvidenceStore, StoreError
        from hyperskill.run_evidence import import_run_directory
        secrets = []
        if Path(args.config).is_file():
            from hyperskill.no_skill_runtime import load_runtime_config
            config = load_runtime_config(args.config)
            secrets = [v for k,v in config.items() if v and any(
                marker in k.upper() for marker in ("API_KEY", "PASSWORD", "SECRET", "TOKEN"))]
        create = args.command in ("init", "import-run", "put-version")
        with EvidenceStore(args.db, secrets=secrets, create=create) as store:
            if args.command in ("init", "summary"):
                output = store.summary()
            elif args.command == "check":
                output = store.integrity_check()
            elif args.command == "import-run":
                output = import_run_directory(store, args.run_dir)
                output["store_summary"] = store.summary()
            elif args.command == "show-task":
                output = store.export_task(args.task_id)
            elif args.command == "show-version":
                output = store.get_skill_version(args.version_id)
            elif args.command == "list-versions":
                output = store.list_skill_versions(args.skill_id)
            else:
                payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
                source_refs = payload.pop("source_refs", [])
                with store.transaction():
                    store.put_skill(payload["skill_id"])
                    version_id = store.put_skill_version(**payload)
                    for ref in source_refs:
                        store.link_version_evidence(version_id=version_id, **ref)
                output = store.get_skill_version(version_id)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        safe = {"type": type(error).__name__}
        if "StoreError" in locals() and isinstance(error, StoreError):
            safe["category"] = error.category
        elif isinstance(error, json.JSONDecodeError):
            safe["category"] = "invalid_json"
        print(json.dumps({"status": "error", "error": safe}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
