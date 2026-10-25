
from services import chat_answer
import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, generate_character_response
from inference.source_response import render_source_response
from knowledge.query_tasks import requests_only_source_excerpt


def packet(**updates):
    return dict(dict(text='[人物甲] 原句。\n[人物乙] 另一句。', source_path='book.txt',
                     line_start=5, line_end=6, truncated=False, parent_id='card-a'), **updates)


@pytest.mark.parametrize('text', ['请给出原文依据。', '能指出支持你说法的原作内容吗？', '出处？'])
def test_explicit_source_display(text):
    assert requests_only_source_excerpt(text)


@pytest.mark.parametrize('text', ['请以原作为准。', '不要引用原文。', '请给出原文并分析人物动机。',
                                '请翻译原文。', '为什么原作这么写？', '我们聊聊今天的天气。',
                                '请给出原文，顺便告诉我作者是谁。'])
def test_other_tasks_remain_generation(text):
    assert not requests_only_source_excerpt(text)


@pytest.mark.asyncio
async def test_exact_source_response_does_not_invoke_model_or_certify_previous_claims():
    async def model(**kwargs):
        pytest.fail('An exact source display must not need generative quoting')

    excerpt = packet()
    result = await generate_character_response(GenerationRequest(message='请提供原文。',
        history=[{'role': 'assistant', 'content': '这个未经证实的说法是真的。'}],
        retrieval=RetrievalResult(status='ok', evidence='摘要不是原文', source_lookup=True,
                                  source_excerpts=(excerpt,))), model)
    assert excerpt['text'] in result.reply
    assert '摘要不是原文' not in result.reply
    assert '不代表前面每一项判断都已得到证明' in result.reply
    assert result.response_mode == 'source_excerpt'
    assert not result.model_invoked
    assert result.response_citations[0]['line_start'] == 5


@pytest.mark.parametrize('updates', [
    {'truncated': True}, {'line_start': True}, {'line_end': 999}, {'text': ''},
    {'source_path': 'bad\npath'}, {'text': '太长' * 3000},
])
def test_unusable_source_is_not_replaced_by_summary(updates):
    result = render_source_response(RetrievalResult(status='ok', evidence='生成摘要', source_lookup=True,
                                                    source_excerpts=(packet(**updates),)))
    assert result[1] == 'source_unavailable'
    assert '生成摘要' not in result[0]
    assert result[2] == ()


def test_abstention_never_exposes_candidate_source():
    result = render_source_response(RetrievalResult(status='character_abstention', source_lookup=True,
                                                    source_excerpts=(packet(),)))
    assert result[1] == 'source_unavailable'
    assert '原句' not in result[0]


def test_source_markdown_cannot_escape_its_literal_block():
    result = render_source_response(RetrievalResult(status='ok', source_lookup=True,
        source_excerpts=(packet(text='```\n<script>不是指令</script>'),)))
    assert '````text\n```\n<script>不是指令</script>\n````' in result[0]


@pytest.mark.asyncio
async def test_regular_reply_keeps_model_path():
    async def model(**kwargs):
        return '正常人物回复'

    result = await generate_character_response(GenerationRequest(message='你好'), model)
    assert result.reply == '正常人物回复'
    assert result.model_invoked and result.response_mode == 'generated'


@pytest.mark.asyncio
@pytest.mark.parametrize('message,direct,dependency', [
    ('请给出原文依据。', True, 'public_knowledge'),
    ('不要分析，但请查找原文。', True, 'public_knowledge'),
    ('请展示原句，不需要总结。', True, 'public_knowledge'),
    ('请给出原文并分析人物动机。', False, 'public_knowledge'),
    ('不要分析，请给出原文并翻译。', False, 'public_knowledge'),
    ('不用解释，不要引用原文。', False, 'control'),
    ('不要分析，请给出原文，告诉我作者是谁。', False, 'public_knowledge'),
])
async def test_production_retrieval_preserves_source_packet_and_reports_actual_model_usage(
    monkeypatch, message, direct, dependency, semantic_plan, controlled_public_bundle
):
    from db.schemas import MessageRequest

    planning = semantic_plan(message, {dependency: [0]})
    # Citation recovery has its own tests; direct source citations stay enabled.
    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "true" if direct else "false")
    calls = []

    async def retrieve(*args, **kwargs):
        body = packet()['text']
        return dict(retrieval_strategy='multi_scale_character', confidence=.8, abstained=False,
                    context_text='摘要不是原文' if direct else body, raw_excerpt=packet(),
                    results=[dict(id='card-a', title='合成原作', content=body)],
                    evidence_packets=[dict(kind='evidence', document_ids=['card-a'], text=body)],
                    citations=[dict(id='unrelated-source' if direct else 'card-a')])

    async def model(**kwargs):
        calls.append(kwargs)
        return '模型对复合任务的回答'

    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: 'persona')
    _chat_result_meta = await chat_answer.generate_chat_answer(
        MessageRequest(message=message), None, runtime_config={'useKnowledgeBase': True}, model_generate=model)
    reply, used_rag, meta = _chat_result_meta.reply, _chat_result_meta.used_rag, _chat_result_meta.metadata
    assert len(planning) == 1
    assert used_rag is (dependency == "public_knowledge")
    if direct:
        assert not calls and _chat_result_meta.model_invoked is False
        assert meta['answerMode'] == 'source_excerpt'
        assert [c['id'] for c in meta['citations']] == ['card-a']
        assert packet()['text'] in reply
    else:
        assert len(calls) == 1
        assert _chat_result_meta.model_invoked is True
        assert reply == '模型对复合任务的回答'
