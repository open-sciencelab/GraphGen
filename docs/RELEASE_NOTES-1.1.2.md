# GraphGen Fork 1.1.2 — event pipeline correctness fixes

Tag: `graphgen-fork-v1.1.2`.

Distribution version: `1.1.2` (`graphg`). Base: `graphgen-fork-v1.1.1` commit `0692702a91d5ea35b468543965998d7877f5700c`.

## Changes

- Preserve explicit filesystem event-profile references through build-service validation so custom project profiles remain loadable by the event builder.
- Resolve support citations using the rendered edge separator and either displayed endpoint order; validate citations only against canonical edges in the partition, while retaining node-only validation for existing callers.
- Skip unanchored events when same-anchor event partitioning is required instead of aborting the partition run.
- Use a JSON-compatible `{` assistant prefill for the opt-in empty schema-extraction retry, and reapply RPM/TPM limits before every actual API attempt.
- Mark event membership edges in direct extraction results and persist their participant role through deterministic edge merging.
- Correct the patch ledger's merge-path description and remove its duplicate concurrency/logger entry.

No default route changes: LightRAG extraction and ECE partitioning remain the default. PMS profiles remain PMS-owned and independently versioned.

## Validation

- GraphGen event, support, profile and package regression tests: see release manifest for commands and results.
- PMS strategy, runner and cache-repair compatibility tests: see release manifest.
- Non-PMS event extraction → star graph → event_join smoke: see release manifest.
- Python 3.12 wheel installation and built-in profile loading: see release manifest.
- Wheel and sdist SHA-256, final commit, profile asset hashes and upstream base are recorded in `RELEASE_MANIFEST-1.1.2.md`.

## Compatibility

- No migration is required for existing configs or graph caches.
- This release only fixes event-profile references, citations, retry accounting, anchor handling and event-member metadata.
- The upstream repository and PR #192 are unchanged.
