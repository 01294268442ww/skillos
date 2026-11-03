skillos — HyperSkill implementation baseline through item 5

This repository imports the upstream HyperSkill working tree and our items 2–5.
Original README.md, package names and task source URLs retain upstream provenance.
SKILLOS_BASELINE.json records the source revision, source hashes and migration scope.

Current design: B+s versus fixed no-skill B, first valid task sets an exact version value,
then that version is frozen. Item 6 valuation has not been implemented. All stored skill
values are still null. An online-value-updating variant has not been implemented.
The current baseline branch is a starting point for later policy branches.

Run on this server:
  cd /home/ubuntu/skillos
  /home/ubuntu/miniconda3/envs/hyperskill-py310/bin/python -m unittest discover -s tests -v
  /home/ubuntu/miniconda3/envs/hyperskill-py310/bin/python runners/evidence_store.py summary

Configuration remains external at /home/ubuntu/.config/hyperskill/runtime.env.
Do not copy credentials into this repository. Existing DeepSeek and Qwen models are reused.
The database storage/evidence.sqlite3 is an independent snapshot; storage/ and results/
are ignored by Git. Historical evidence contains the original source paths intentionally.
Future value-policy experiments should use separate Git branches/worktrees AND separate
storage/evidence.sqlite3 files so their observations and value records remain independent.

The original /home/ubuntu/HyperSkill working tree and database are preserved.
No skill valuation, policy change, task execution or remote push is part of this migration.
