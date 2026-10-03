"""Complete independent compound read tasks, not private native fixtures."""

import pytest

from knowledge.source_expansion import requested_document_titles


@pytest.mark.parametrize(
    "closing", ["本轮只读取，不新增或删除记忆。", "本轮只读取， 不新增或删除记忆。", "本轮只读取，\n不新增或删除记忆。"]
)
def test_closed_nonmutating_memory_scope_retains_declared_originals(closing):
    query = (
        "读取《紫茶公共办理规则》《紫茶合成角色版本》。分别说明公定费用、完整条件、失败处理和预约例外；本人当前与未来条件只依据保存原话，角色与本人主体分开，未知实际办理明确未知。"
        + closing
    )
    assert requested_document_titles(query) == ("紫茶公共办理规则", "紫茶合成角色版本")


@pytest.mark.parametrize(
    "closing",
    [
        "本轮只读取第一份，不新增或删除记忆。",
        "本轮只读取《另一资料》，不新增或删除记忆。",
        "本轮只读取，不新增或删除记忆，但不要读取第一份。",
        "这里只转述“本轮只读取，不新增或删除记忆。”",
        "本轮只读取，不新增或删除记忆中的第一条。",
        "本轮只读取，不新增或删除记忆。另读取《另一资料》。",
    ],
)
def test_document_exclusions_unknown_targets_and_quotes_are_not_memory_scope(closing):
    query = "读取《紫茶公共办理规则》《紫茶合成角色版本》。请核对全部说明。" + closing
    assert requested_document_titles(query) == ()
