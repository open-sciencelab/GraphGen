# GraphGen Fork 1.1.0 — production runtime fixes

Tag: `graphgen-fork-v1.1.0`.

Distribution version: `1.1.0` (`graphg`). Base: `graphgen-fork-v1.0.0` commit `c5ef453aa661776b757902a990c529fdab2bfc9a`.

## Changes

- Added bounded, opt-in Ray LLM actor concurrency through `GRAPHGEN_LLM_ACTOR_MAX_CONCURRENCY`. The default remains Ray's existing concurrency of 1; values greater than 1 are passed as the actor's `max_concurrency` option.
- Logger calls from Ray contexts without a propagated logger context now use the `GraphGen` standard logger instead of raising during logging. This fallback changes logging only; model errors and generated output semantics are unchanged.
- PMS prompt policy remains an independently versioned consumer policy. This release does not overwrite the immutable v1.0.0 prompt assets.

## Validation

- New actor concurrency and logger context tests: 4 passed.
- Event profile and package tests: 2 passed when combined with the new unit tests.
- Wheel and sdist built; each contains actor and logger modules plus all four generation profile JSON files.
- Clean Python 3.12 venv installed the wheel. `graphg` reported 1.1.0; both built-in PMS policy profiles loaded; the `pms_project_anchor` output gate worked.
- Wheel and sdist SHA-256 values are recorded alongside the exact release commit in the release artifact manifest.

PMS full production integration is validated separately in the PMS repository, which pins this fork commit explicitly and owns PMS policy files.
