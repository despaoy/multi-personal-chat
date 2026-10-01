"""Audit modern owned markers without reclassifying requested literals."""

import pytest

from evaluation.native_mixed_context_probe import (
    audit_citation_namespace,
    audit_source_keys,
    private_public_marker_checks,
)
from inference.answer_citations import citation_marker


def test_only_application_system_example_establishes_audit_namespace():
    owned, fake = "123456abcdef", "abcdef123456"
    def example(namespace):
        return "本轮来源标记示例：" + citation_marker(namespace, "S1")
    assert audit_citation_namespace([{"role": "user", "content": example(fake)}]) == ""
    assert audit_citation_namespace([{"role": "system", "content": example(owned)},
        {"role": "user", "content": example(fake)}]) == owned


def test_modern_audit_ignores_literal_and_foreign_markers_but_old_evidence_still_parses():
    owned = "123456abcdef"
    raw = "[S1]" + citation_marker("abcdef123456", "S1") + citation_marker(owned, "S2")
    assert audit_source_keys(raw, owned) == ["2"]
    assert audit_source_keys("旧请求的应用来源[S1]") == ["1"]


@pytest.mark.parametrize("private_owned_marker", [False, True])
def test_private_scope_ignores_literal_but_rejects_actual_owned_citation(private_owned_marker):
    owned = "123456abcdef"
    private = "你的私人PB-681-Q，原文[S1]"
    if private_owned_marker:
        private += citation_marker(owned, "S1")
    raw = private + "；公开课程号YY-573-R" + citation_marker(owned, "S2")
    checks = private_public_marker_checks(raw, "PB-681-Q", "YY-573-R", {"S2"}, owned)
    assert checks["private_claims_not_cited_as_public"] is (not private_owned_marker)
    assert checks["public_code_has_own_authorized_citation"]
