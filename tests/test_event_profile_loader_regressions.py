from __future__ import annotations

import json

import pytest

from graphgen.templates.event_profile_loader import load_event_profile


def _copy_profile(root, profile_id: str = "custom_event.v1"):
    source = load_event_profile("example_event.v1")
    manifest = {
        "schema_version": "graphgen_event_profile.v1",
        "profile_id": profile_id,
        "prompt_file": "prompt.zh.txt",
        "entity_types": list(source.entity_types),
        "anchor": {"required": False, "pattern": None, "entity_type": None},
        "grounding": {"number_grounding": False},
        "graph": {"event_entity_type": "EVENT", "legacy_event_id": False},
    }
    root.mkdir(parents=True)
    (root / "profile.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "prompt.zh.txt").write_text("{input_text}", encoding="utf-8")
    return root / "profile.json"


def test_event_member_role_survives_deterministic_edge_merge():
    import asyncio

    from graphgen.models.kg_builder.event_entity_kg_builder import EventEntityKGBuilder

    class Merger:
        tokenizer = type("Tokenizer", (), {"count_tokens": lambda _self, text: len(text)})()

        async def merge_edges(self, *_args, **_kwargs):
            raise AssertionError("event membership edge must not use LightRAG summary")

    class Storage:
        def __init__(self):
            self.edge_data = None

        def get_edge(self, *_args):
            return None

        def upsert_edge(self, _src, _tgt, edge_data):
            self.edge_data = edge_data

    builder = EventEntityKGBuilder.__new__(EventEntityKGBuilder)
    builder.profile = load_event_profile("example_event.v1")
    builder._merger = Merger()
    storage = Storage()
    edge = {
        "src_id": "event:case-1",
        "tgt_id": "Operator",
        "description": "participated",
        "source_id": "chunk-1",
        "role": "ACTOR",
        "event_member": True,
    }

    merged = asyncio.run(
        builder.merge_edges((("event:case-1", "Operator"), [edge]), storage)
    )

    assert merged["role"] == "ACTOR"
    assert storage.edge_data["role"] == "ACTOR"
def test_filesystem_event_profile_reference_can_be_reloaded(tmp_path):
    """The service-normalized reference preserves the absolute manifest path."""
    manifest = _copy_profile(tmp_path / "profiles" / "custom_event.v1")
    reference = str(manifest.resolve())

    assert reference == str(manifest.resolve())
    assert load_event_profile(reference).prompt == "{input_text}"
    assert load_event_profile(reference).prompt == load_event_profile(reference).prompt


def test_build_kg_service_preserves_filesystem_event_profile_path(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path

    from graphgen.templates.event_profile_loader import load_event_profile

    service_module = Path(__file__).parents[1] / "graphgen/operators/build_kg/build_kg_service.py"
    package_name = "graphgen.operators.build_kg.build_kg_service_regression"
    fake_bases = types.ModuleType("graphgen.bases")
    fake_bases.BaseGraphStorage = type("BaseGraphStorage", (), {})
    fake_bases.BaseLLMWrapper = type("BaseLLMWrapper", (), {})
    fake_bases.BaseOperator = type("BaseOperator", (), {"__init__": lambda self, **_kwargs: None})
    fake_datatypes = types.ModuleType("graphgen.bases.datatypes")
    fake_datatypes.Chunk = type("Chunk", (), {})
    monkeypatch.setitem(sys.modules, "graphgen.bases", fake_bases)
    monkeypatch.setitem(sys.modules, "graphgen.bases.datatypes", fake_datatypes)
    monkeypatch.setitem(sys.modules, "graphgen.common.init_llm", types.SimpleNamespace(init_llm=lambda *_args: object()))
    monkeypatch.setitem(sys.modules, "graphgen.common.init_storage", types.SimpleNamespace(init_storage=lambda **_kwargs: object()))
    monkeypatch.setitem(sys.modules, "graphgen.utils", types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_a: None)))
    captured_event_profiles = []

    def capture_event_profile(**kwargs):
        captured_event_profiles.append(kwargs["event_profile"])
        return [], []

    monkeypatch.setitem(
        sys.modules,
        "graphgen.operators.build_kg.build_event_kg",
        types.SimpleNamespace(build_event_entity_kg=capture_event_profile),
    )
    monkeypatch.setitem(sys.modules, "graphgen.operators.build_kg.build_mm_kg", types.SimpleNamespace(build_mm_kg=lambda **_kwargs: ([], [])))
    monkeypatch.setitem(sys.modules, "graphgen.operators.build_kg.build_text_kg", types.SimpleNamespace(build_text_kg=lambda **_kwargs: ([], [])))
    spec = importlib.util.spec_from_file_location(package_name, service_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = _copy_profile(tmp_path / "profiles" / "custom_event.v1")
    reference = str(manifest.resolve())
    service = module.BuildKGService.__new__(module.BuildKGService)
    module.BuildKGService.__init__(service, kg_method="event_entity", event_profile=reference)
    assert service.event_profile == reference
    assert load_event_profile(service.event_profile).prompt == "{input_text}"
    service.llm_client = object()
    service.graph_storage = types.SimpleNamespace(index_done_callback=lambda: None)
    fake_datatypes.Chunk.from_dict = lambda _trace_id, doc: types.SimpleNamespace(
        type=doc["type"], content=doc["content"]
    )
    service.process(
        [{"_trace_id": "chunk-1", "type": "text", "content": "project update"}]
    )
    assert captured_event_profiles == [reference]

    builtin = module.BuildKGService.__new__(module.BuildKGService)
    module.BuildKGService.__init__(builtin, kg_method="event_entity", event_profile="example_event.v1")
    assert builtin.event_profile == "example_event.v1"
    invalid = module.BuildKGService.__new__(module.BuildKGService)
    with pytest.raises(ValueError, match="event_profile_not_found"):
        module.BuildKGService.__init__(invalid, kg_method="event_entity", event_profile="/missing/profile.json")


def test_event_support_edge_alias_matches_rendered_separator():
    from graphgen.models.generator.support import build_generation_view, validate_support

    event_id = "event:opaque-hash"
    item = {
        "nodes": [
            (event_id, {"entity_type": "EVENT", "graph_role": "event", "description": "Migration completed：fact"}),
            ("Operator", {"entity_type": "ACTOR", "description": "responsible"}),
        ],
        "edges": [(event_id, "Operator", {"description": "participated"})],
    }
    view = build_generation_view(item)
    citation = f'{view["nodes"][0][0]} -- Operator'

    resolved = view["resolve_support"]({"cited": [citation]})

    assert resolved == {"cited": [f"{event_id} - Operator"]}
    assert validate_support(
        resolved,
        set(view["canonical_nodes"]),
        set(view["canonical_edges"]),
    ) == resolved


def test_event_join_skips_missing_anchor_and_yields_valid_anchored_event():
    from graphgen.models.partitioner.event_join_partitioner import EventJoinPartitioner

    class FakeStorage:
        def get_all_nodes(self):
            return [
                ("event:unanchored", {"entity_type": "EVENT", "graph_role": "event", "anchor": "", "length": 1}),
                ("event:anchored", {"entity_type": "EVENT", "graph_role": "event", "anchor": "CASE-1", "length": 1}),
                ("entity:operator", {"entity_type": "ACTOR", "length": 1}),
            ]

        def get_neighbors(self, node_id):
            return ["entity:operator"] if node_id == "event:anchored" else []

    communities = list(EventJoinPartitioner().partition(FakeStorage(), require_same_anchor=True))

    assert len(communities) == 1
    assert communities[0].id == "event:anchored"
    assert "event:unanchored" not in communities[0].nodes




def test_schema_guided_extractor_uses_json_compatible_prefill(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from graphgen.bases.datatypes import Chunk
    from graphgen.models.extractor.schema_guided_extractor import SchemaGuidedExtractor

    class FakeLLM:
        def __init__(self):
            self.calls = []

        async def generate_answer(self, text, **kwargs):
            self.calls.append(kwargs.get("assistant_prefill"))
            return "" if len(self.calls) == 1 else '{"name":"value"}'

    monkeypatch.setenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", "1")
    llm = FakeLLM()
    extractor = SchemaGuidedExtractor(llm, {"name": "item", "properties": {"name": {"description": "name"}}, "required": ["name"]})
    result = asyncio.run(extractor.extract(Chunk(id="chunk-1", content="a short text", type="text")))

    assert result == {"name": "value"}
    assert llm.calls == [None, "{"]


def test_openai_empty_retry_reapplies_rpm_and_tpm_limits(monkeypatch):
    import asyncio
    import types

    from graphgen.models.llm.api.openai_client import OpenAIClient

    class Limiter:
        def __init__(self):
            self.calls = []

        async def wait(self, *args, **kwargs):
            self.calls.append(args)

    class Completion:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    class Completions:
        def __init__(self):
            self.responses = iter([Completion(""), Completion("ok")])

        async def create(self, **_kwargs):
            return next(self.responses)

    client = object.__new__(OpenAIClient)
    client._pre_generate = lambda *_args, **_kwargs: {"messages": [{"content": "x"}], "max_tokens": 4}
    client.tokenizer = types.SimpleNamespace(encode=lambda _text: [1])
    client.request_limit = True
    client.rpm, client.tpm = Limiter(), Limiter()
    client.model = "test"
    client.token_usage = []
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    monkeypatch.setenv("GRAPHGEN_EMPTY_RETRY", "1")

    assert asyncio.run(client.generate_answer("prompt")) == "ok"
    assert len(client.rpm.calls) == 2
    assert len(client.tpm.calls) == 2
