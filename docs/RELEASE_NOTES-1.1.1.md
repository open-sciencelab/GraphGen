# GraphGen Fork 1.1.1 — event extraction prompt fix

Tag: `graphgen-fork-v1.1.1`.

Distribution version: `1.1.1` (`graphg`). Base: `graphgen-fork-v1.1.0` commit `e11158cd8254f98c827690b60648373149f8a8f5`.

## Changes

- Built-in `pms_event.v1` extraction prompt now forbids copying the example project id: the anchor must be taken verbatim from the graph fragment. This mirrors the generation-policy fix that removed the previously observed defect where example ids leaked into production output.
- No API, schema, or default-route change. LightRAG + ECE remains the default.

## Validation

- GraphGen unit, event profile, and package tests pass in the pinned environment.
- Wheel and sdist are built and inspected; the fixed prompt ships inside both.
- Consumer verification (install, profile load, end-to-end extraction and generation) is recorded in the PMS repository that pins this commit.
