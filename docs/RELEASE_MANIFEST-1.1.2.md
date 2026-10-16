# GraphGen 1.1.2 release manifest

Status: validation evidence for the release candidate; commit SHA is recorded after final release commit.

## Source identity

- Distribution: `graphg` 1.1.2
- Candidate base/tag: `graphgen-fork-v1.1.1` / `0692702a91d5ea35b468543965998d7877f5700c`
- Upstream base: `origin/main` `3a3eb097318d34b07f6f31cb60de5b20712d65ce`
- Intended publication remote: `fork` (`https://github.com/Leon-Algo/GraphGen.git`); upstream `origin` is not to be modified.
- PMS consumer policy remains PMS-owned; no PMS prompt files are included in the GraphGen diff.

## Test and smoke evidence

Run in the GraphGen defect worktree with Python 3.12 dependencies supplied by `workspace/graphgen_env/.venv`; `PYTHONPATH` must put the defect worktree before the installed wheel when testing source changes.

- `PYTHONPATH=<defect-worktree>:<graphgen-env-site-packages> UV_OFFLINE=1 NLTK_DISABLE_IMPORT_SECURITY=1 <root-venv-python> -m pytest -c /dev/null -q tests/unit_tests tests/test_event_profile_loader_regressions.py tests/test_support_edge_citations.py tests/test_event_profiles.py` — **15 passed**.
- `PYTHONPATH=<defect-worktree>:<graphgen-env-site-packages> UV_OFFLINE=1 NLTK_DISABLE_IMPORT_SECURITY=1 <root-venv-python> -m pytest -c /dev/null -q tests/test_package_profiles.py` — **1 passed**.
- `PYTHONPATH=<defect-worktree>:<graphgen-env-site-packages> NLTK_DISABLE_IMPORT_SECURITY=1 <root-venv-python> -m pytest -c /dev/null -q <PMS>/tests/test_pms_graphgen_policy_generation.py <PMS>/tests/test_run_graphgen_defaults.py <PMS>/tests/test_repair_project_query_graphgen_chunk_cache.py` — **28 passed, 1 skipped**.
- `PYTHONPATH=<defect-worktree>:<graphgen-env-site-packages> NLTK_DISABLE_IMPORT_SECURITY=1 <root-venv-python> -m pytest -c /dev/null -q tests/integration_tests/models/partitioner/test_ece_partitioner.py` — **6 passed**.
- Non-PMS deterministic integration smoke via `EventEntityKGBuilder` extraction+merge → `EventJoinPartitioner`: **3 stub LLM calls, 3 event nodes, 3 membership edges, 3 communities; role persisted on every edge; anchors CASE-A/CASE-B/CASE-C; no network/model calls**. Executed by the GraphGen test agent; command: `PYTHONPATH=<defect-worktree> <graphgen-env>/.venv/bin/python /tmp/graphgen_event_join_smoke.py`.
- `git diff --check` — passed.

## Build and clean-install evidence

Built in a disposable output directory with `uv build --offline --out-dir /tmp/graphgen-release-1.1.2`:

- `graphg-1.1.2-py3-none-any.whl` — SHA-256 `5272d7c6a03eaa3d8add95b2674946a0f20895a9a420e3f547ee04a54477f792`
- `graphg-1.1.2.tar.gz` — SHA-256 `234d0ec6302a92077fda5e9edfd35b542784de4bf50841e6dab3ccef5474cab3`
- Archive inspection confirmed `profile.json` and `prompt.zh.txt` for `example_event.v1`, `generic_event.v1`, and `pms_event.v1` in both artifacts.
- Installed the wheel with `uv pip install --offline --python /tmp/graphgen-clean312/bin/python --no-deps --force-reinstall <wheel>` into a fresh Python 3.12 venv. Confirmed `graphg==1.1.2`, `graphgen.__file__` under that venv's `site-packages`, and loaded all three built-in event profiles. Prompt hashes:
  - `example_event.v1`: `8a27caf4453f9734a499a63c97ee68c5cb0d73297409df86a6d82461af31c828`
  - `generic_event.v1`: `801aa5cd8967d608dbdcf5d9057fe7c91e9154862516be85944c335207750136`
  - `pms_event.v1`: `7ab5656e8b80a90310a1a7645ae0b11b752127a5628d1b3de2994d9929552b25`

## Publication record

- Final commit SHA: pending
- Tag creation: pending; must verify local and `fork` remote tag absence first.
- Fork branch/tag push: pending maintainer action after final diff review.
- PMS dependency pin update and locked sync: pending 1.1.2 commit SHA after publication.
