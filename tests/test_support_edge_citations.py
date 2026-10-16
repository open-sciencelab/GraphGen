from graphgen.models.generator.support import build_generation_view, validate_support


def _item():
    event_id = "event:opaque-hash"
    return {
        "nodes": [
            (event_id, {"entity_type": "EVENT", "graph_role": "event", "description": "Migration completed：fact"}),
            ("Operator", {"entity_type": "ACTOR", "description": "responsible"}),
        ],
        "edges": [(event_id, "Operator", {"description": "participated"})],
    }


def test_rendered_edge_alias_resolves_and_only_present_canonical_edges_validate():
    item = _item()
    view = build_generation_view(item)
    citation = f'{view["nodes"][0][0]} -- Operator'
    resolved = view["resolve_support"]({"cited": [citation]})
    known_edge = "event:opaque-hash - Operator"

    assert resolved == {"cited": [known_edge]}
    assert validate_support(
        resolved,
        set(view["canonical_nodes"]),
        set(view["canonical_edges"]),
    ) == resolved
    assert validate_support(
        {"cited": ["event:opaque-hash - Missing"]},
        set(view["canonical_nodes"]),
        set(view["canonical_edges"]),
    ) is None


def test_rendered_edge_alias_resolves_when_partition_reverses_edge_endpoints():
    item = _item()
    event_id = item["nodes"][0][0]
    item["edges"] = [("Operator", event_id, {"description": "participated"})]
    view = build_generation_view(item)
    citation = f'Operator -- {view["nodes"][0][0]}'

    resolved = view["resolve_support"]({"cited": [citation]})

    assert resolved == {"cited": [f"Operator - {event_id}"]}
    assert validate_support(
        resolved,
        set(view["canonical_nodes"]),
        set(view["canonical_edges"]),
    ) == resolved
