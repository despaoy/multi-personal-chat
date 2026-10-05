"""Distinct complete fictional registries may both deny current identity.

These exercise receipt parsing, not model judgement or native authorization.
"""

import json
from copy import deepcopy

import pytest

from knowledge.public_identity_dependencies import parse_identity_review
from knowledge.public_object_scope import parse_object_scopes

QUERY = "核对霁岚换证当前费用与例外。另核对晴汀续领的完整规则。"
OLD = "旧版名称登记，现已作废，仅供历史追溯：霁岚换证与铜湾补领曾登记为同一业务。旧版不证明当前同一性，不提供当前规则。"
CURRENT = "当前有效新版名称登记：霁岚换证不是铜湾补领，二者当前是不同业务。旧版已废止，本页仅登记名称关系，不提供当前办理规则。"
DIRECT = (
    "晴汀续领收费26元，需要2个工作日，每周四15:05截止；登记证原件和收件表原件必需，暴雨天暂停，预约不能豁免这两项原件。"
)


def fixture():
    sources = [
        dict(
            source_id=f"doc_{i}",
            title=f"独立虚构原文{i}",
            knowledge_base_id=7,
            indexed_chunks=[dict(id=f"doc_{i}_chunk_0", content=body)],
            original_body=body,
        )
        for i, body in enumerate([OLD, CURRENT, DIRECT], 1)
    ]
    payload = dict(query=QUERY, public_task_ids=[0, 1], sources=sources)
    scopes = parse_object_scopes(
        json.dumps(dict(scopes=[dict(task_id=0, objects=["霁岚换证"]), dict(task_id=1, objects=["晴汀续领"])])),
        QUERY,
        [0, 1],
    )
    data = dict(
        sources=[
            dict(source_id="doc_1", purpose="identity_only"),
            dict(source_id="doc_2", purpose="identity_only"),
            dict(source_id="doc_3", purpose="rules"),
        ],
        relations=[
            dict(
                object_id="query-object:0", alias="铜湾补领", source_id="doc_1", source_quote=OLD, relation="uncertain"
            ),
            dict(
                object_id="query-object:0",
                alias="铜湾补领",
                source_id="doc_2",
                source_quote=CURRENT,
                relation="different",
            ),
        ],
    )
    return payload, scopes, data


@pytest.mark.parametrize("reverse", [False, True])
def test_expired_positive_and_current_negative_keep_distinct_source_assessments(reverse):
    payload, scopes, data = fixture()
    if reverse:
        data["relations"].reverse()
    result = parse_identity_review(json.dumps(data), payload, scopes)
    assert result["bindings"] == []
    assert result["source_roles"] == data["sources"]
    assert payload["sources"][2]["original_body"] == DIRECT
    assert data["relations"][0]["source_id"] != data["relations"][1]["source_id"]


@pytest.mark.parametrize("other", ["different", "uncertain"])
def test_affirmative_conflict_cannot_become_authority(other):
    payload, scopes, data = fixture()
    data["relations"][0]["relation"] = "same"
    data["relations"][1]["relation"] = other
    with pytest.raises(ValueError, match="Conflicting identity relation"):
        parse_identity_review(json.dumps(data), payload, scopes)


def test_repeated_nonaffirmative_assessment_from_one_source_is_rejected():
    payload, scopes, data = fixture()
    data["relations"].append(deepcopy(data["relations"][0]))
    with pytest.raises(ValueError, match="Conflicting identity relation"):
        parse_identity_review(json.dumps(data), payload, scopes)


def test_distinct_nonaffirmative_sources_still_require_literal_complete_evidence():
    payload, scopes, data = fixture()
    data["relations"][1]["source_quote"] += "当前费用为29元。"
    with pytest.raises(ValueError, match="literal complete source evidence"):
        parse_identity_review(json.dumps(data), payload, scopes)
