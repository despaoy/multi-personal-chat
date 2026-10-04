"""A nonmutating existing-data scope is not a document exclusion selector."""

import pytest

from knowledge.source_expansion import requested_document_titles

PREFIX = "请分别读取《青川规程》《蓝溪规程》并分别核对规则。"
EXPECTED = ("青川规程", "蓝溪规程")


def baseline():
    assert requested_document_titles(PREFIX + "本轮只读取，不新增或删除记忆。") == EXPECTED


@pytest.mark.parametrize("scope", ["既有事实", "已有事实", "既有资料", "已有信息"])
def test_generic_existing_data_scope_keeps_all_independent_roots(scope):
    baseline()
    assert requested_document_titles(PREFIX + f"本轮只读取{scope}，不新增或删除记忆，不重放旧目录任务。") == EXPECTED


@pytest.mark.parametrize(
    "suffix",
    [
        "本轮只读取第一份，不新增或删除记忆。",
        "本轮只读取《青川规程》，不新增或删除记忆。",
        "本轮只读取既有事实。",
        "本轮只读取既有事实，不新增或删除记忆，但不要读取《蓝溪规程》。",
        "本轮只读取既有事实，不新增或删除记忆，《白石规程》也要核对。",
    ],
)
def test_exclusions_and_unclosed_new_roots_are_never_erased(suffix):
    baseline()
    assert requested_document_titles(PREFIX + suffix) == ()


def test_quoted_background_cannot_supply_the_nonmutation_clause():
    baseline()
    assert (
        requested_document_titles(PREFIX + "本轮只读取第一份。背景原话：“本轮只读取既有事实，不新增或删除记忆”。") == ()
    )
