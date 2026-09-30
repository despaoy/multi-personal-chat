"""Bounded runtime operation facts, without source text or model instructions."""

import re


def is_only_memory_erasure(message: str) -> bool:
    """A bounded imperative grammar, not a classifier for arbitrary tasks.

    Ambiguous/mixed requests keep the normal response path. This gate never
    authorizes a mutation; it only selects rendering after an actual receipt.
    """
    from character.memory_llm import is_memory_erasure_request

    if not is_memory_erasure_request(message):
        return False
    parts = [part.strip() for part in re.split(r'[，,。；;！!\n]', message) if part.strip()]
    if not parts:
        return False
    for index, part in enumerate(parts):
        if index and re.fullmatch(r'(?:也)?(?:不要|别)再(?:引用|提及|提起)(?:那条原话|这条原话|它|这件事)', part):
            continue
        # Independent task connectors and interrogatives cannot be swallowed
        # as an arbitrary deletion object's name.
        if re.search(r'并|然后|顺便|同时|另外|再|以及|告诉|解释|为什么|怎么|[？?]', part):
            return False
        if not re.fullmatch(
            r'(?:请)?(?:(?:把|将).{1,120}?(?:删除|删掉|清除|清空|忘掉)'
            r'|(?:从(?:长期)?记忆[里中])?(?:删除|删掉|清除|忘掉|忘记).{1,80})', part
        ):
            return False
    return True


def render_operation_response(message: str, receipt: object) -> str | None:
    if not isinstance(receipt, dict) or not is_only_memory_erasure(message):
        return None
    status = receipt.get('status')
    count = receipt.get('persisted')
    if status == 'erased' and type(count) is int and count > 0:
        if receipt.get('source_erased', 0):
            return '已删除本次匹配的记录；未逐一核对全部历史，聊天存档没有删除。'
        return '已删除匹配的长期记忆及关联来源；聊天记录没有删除。'
    return {
        'pending': '删除请求已受理，还在处理中，暂时不能确认完成。',
        'partial': '这次只完成了部分记忆操作，尚未全部完成。',
        'cancelled': '操作中途被中断，最终结果还不能确认。',
        'conflict': '记录在处理期间发生了变化，这次删除未完成。',
        'not_scheduled': '这次删除请求未能受理，还没有执行。',
        'failed': '这次记忆操作未能完成。',
        'skipped': '这次没有执行删除。',
        'no_change': '这次没有删除任何记录，尚不能确认你指定的信息已清除。',
        'saved': '这次有记忆写入，但没有已确认的删除结果。',
    }.get(status if isinstance(status, str) else '', '这次删除的结果尚未确认。')


def split_operation_request(message: str) -> tuple[str, str] | None:
    """Split an explicit leading operation from a separate remaining task.

    Keep the entire remainder, including punctuation and additional tasks.
    Quoted text and questions about the operation itself remain opaque.
    """
    if any(mark in message for mark in ('"', '“', '”', '「', '」', '『', '』')):
        return None
    if is_only_memory_erasure(message):
        return None
    for boundary in re.finditer(r'[，,。；;\n]\s*(?:(?:同时|另外|顺便|然后|并且|并)\s*)?|(?:同时|另外|顺便|然后|并且)', message):
        operation = message[:boundary.start()].strip()
        remaining = message[boundary.end():].strip()
        if not remaining or not is_only_memory_erasure(operation):
            continue
        # Do not isolate follow-ups whose referent/condition is the mutation.
        if re.search(r'删除|删掉|忘掉|记忆|(?:刚才|上述|这次|这些|那条|它|结果)|(?:成功|失败|完成)后', remaining):
            return None
        return operation, remaining
    return None

_STATUS_TEXT = {
    'erased': '已确认执行删除。',
    'saved': '已确认写入，但没有已确认的删除操作。',
    'partial': '仅部分操作完成，不能视为全部完成。',
    'no_change': '没有执行任何已确认的记忆变更，不能据此声称已删除。',
    'not_scheduled': '本轮操作未受理，未执行删除。',
    'pending': '已受理，等待执行结果；当前尚未确认删除。',
    'cancelled': '执行被中断，最终结果未确认；不代表全部未执行。',
    'conflict': '记忆版本已变化，本轮提案没有完成。',
    'skipped': '本轮操作未执行。',
    'failed': '操作未成功完成。',
}


def operation_receipt_context(receipt: dict[str, object]) -> str:
    """Only fixed status facts enter the system context, never raw errors."""
    status = receipt.get('status')
    text = _STATUS_TEXT.get(status, '操作结果未确认。') if isinstance(status, str) else '操作结果未确认。'
    count = receipt.get('persisted', 0)
    count = count if type(count) is int and 0 <= count <= 100 else 0
    return f'【本轮长期记忆操作执行回执】\n{text}\n已确认完成的操作数：{count}。\n范围仅为长期记忆及关联来源，不表示聊天历史已经删除。'
