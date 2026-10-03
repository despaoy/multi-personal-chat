"""Independent whole-source wire examples, never used as native seeds."""

import json
from html import escape

import pytest

from evaluation.source_transport import complete_source_in_transport

SOURCE = (
    "本人合成陈述。\n从2028年5月1日起，\t我只在登记页已提交且紫色封签核验通过时选择紫茶。\n“旁人喜欢绿茶”只作引述。"
)


@pytest.mark.parametrize("form", ["plain", "json_scalar", "json_array", "html_json"])
def test_complete_original_survives_native_quote_transport(form):
    transport = (
        SOURCE
        if form == "plain"
        else json.dumps(SOURCE if form == "json_scalar" else [SOURCE, "单独证据片段"], ensure_ascii=False)
    )
    if form == "html_json":
        transport = escape(transport)
    assert complete_source_in_transport(transport, SOURCE)


@pytest.mark.parametrize("change", ["fragment", "normalized", "changed_condition", "reordered", "truncated", "other"])
def test_fragments_paraphrases_and_different_conditions_are_not_complete(change):
    text = {
        "fragment": "登记页已提交且紫色封签核验通过",
        "normalized": "".join(SOURCE.split()),
        "changed_condition": SOURCE.replace("且", "或"),
        "reordered": "\n".join(reversed(SOURCE.splitlines())),
        "truncated": SOURCE[:-1],
        "other": "完整的另一个主体原话，不是这份原文。",
    }[change]
    assert not complete_source_in_transport(json.dumps([text], ensure_ascii=False), SOURCE)


@pytest.mark.parametrize("source", ["", "  ", None])
def test_empty_expected_source_is_not_an_absence_waiver(source):
    with pytest.raises(ValueError):
        complete_source_in_transport("anything", source)
