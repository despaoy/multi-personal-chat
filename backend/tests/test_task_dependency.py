import pytest

from knowledge.intent_detector import RAGIntentDetector
from knowledge.task_dependency import local_context_only


@pytest.mark.parametrize('message', [
    '请记住，我的专业是材料学。', '请记住我住在兰州。',
    '记下：我叫文清。', '请记住，我喜欢古典音乐。',
    '你还保存着我的专业吗？', '你现在还保存着我的住址吗？',
    '你目前有没有记录我的工作地点？', '你已经记住我的姓名和专业了吗？',
])
def test_memory_management_has_no_external_dependency(message):
    assert local_context_only(message)
    assert RAGIntentDetector().needs_rag(message)[0] is False


@pytest.mark.parametrize('message', [
    '请记住我的专业是材料学，同时解释什么是晶体缺陷。',
    '请记住，我住在兰州。请查一下明天的天气。',
    '请记住我叫什么，另外月社妃和琉璃是什么关系？',
    '你还保存着我的专业吗？请介绍材料学的发展。',
    '你有没有记录我的专业以及推荐专业教材？',
    '请记住月社妃的哥哥是谁，给出原文。',
    '你还保存着我的专业的课程大纲吗？',
])
def test_management_wrapper_does_not_hide_external_or_unknown_task(message):
    assert not local_context_only(message)


@pytest.mark.parametrize('message', [
    '不用给我建议，我只是想说说今天的事。',
    '不要分析，我只想聊聊自己的感受。',
    '别再追问了，我只想说说我的经历。',
    '请不要给我建议。', '我只是想讲讲最近的事。',
    '陪我聊聊天。', '听我说说就好。',
])
def test_conversation_preferences_are_not_external_information_requests(message):
    assert local_context_only(message)
    assert RAGIntentDetector().needs_rag(message)[0] is False


@pytest.mark.parametrize('message', [
    '不用给我建议，请解释量子纠缠。',
    '我只想聊聊自己的感受，月社妃是谁？',
    '不要分析，但请查找原文。',
    '不用给我建议我想知道地球的年龄',
    '我只是想说说今天的事并请解释数据库索引',
    '我想听你讲讲第二次世界大战。',
    '我只想聊聊量子力学。',
    '不要给我建议是什么含义？',
    '不是不要分析，是请你分析原因。',
    '听我说说就好，顺便列举相关研究。',
])
def test_local_conversation_clause_does_not_hide_external_dependencies(message):
    assert not local_context_only(message)
    assert RAGIntentDetector().needs_rag(message)[0] is True


@pytest.mark.parametrize('message', [
    '我只有场地获批才举办活动。', '我只有充分休息才开车。',
    '我只有完成工作才看电影。', '我只有设备检查通过才启动仪器。',
    '更正一下，我只有收到确认才预约会议。',
])
def test_complete_self_owned_constraints_need_no_external_retrieval(message):
    assert local_context_only(message)
    assert RAGIntentDetector().needs_rag(message)[0] is False


@pytest.mark.parametrize('message', [
    '我只有场地获批才举办活动，请介绍审批流程。',
    '我只有完成工作才看电影帮我推荐电影',
    '我只有设备检查通过才启动仪器，这种设备原理是什么？',
    '我只有收到确认才预约会议以及谁负责审批',
    '我只有读过原文才相信你，给出原文依据。',
    '我只有充分休息才开车吗？',
    '我只有收到确认才想了解审批流程',
    '如果我只有场地获批才举办活动，那么需要哪些手续？',
    '她只有完成工作才看电影。',
])
def test_constraints_do_not_swallow_external_or_unknown_tasks(message):
    assert not local_context_only(message)


@pytest.mark.parametrize('message', [
    '今天没有什么特别的事情。', '我刚学会一个新方法。', '我最近在看一些历史书。',
    '我的专业是统计学，现在在气象站工作。', '我不怎么喜欢这个颜色。',
    '我喜欢什么音乐？', '你还记得我叫什么吗？', '我最近要完成什么？',
    '我的专业和工作地点是什么？',
    '我还没告诉你我叫什么。', '我没有告诉过你我的专业是什么。',
    '我一直没有向你提过我喜欢什么音乐。',
])
def test_topics_and_non_specific_words_do_not_create_external_tasks(message):
    assert local_context_only(message)
    assert RAGIntentDetector().needs_rag(message)[0] is False


@pytest.mark.parametrize('message', [
    '我喜欢什么音乐？再介绍一下爵士乐。', '我最近要完成什么？解释一下回声原理。',
    '我今天没什么事，为什么会出现极光？', '我刚学了统计，请解释贝叶斯定理。',
    '今天的气象数据怎么看？', '月社妃和夜子是什么关系？',
    '你还记得我吗？解释一下数据库索引。', '我喜欢什么音乐，帮我分析交响乐的结构。',
    '我最近要学习什么是深度学习？', '我喜欢什么音乐以及爵士乐的起源是什么？',
    '我想了解一下量子力学。', '今天晴不晴？', '我喜欢什么样的音乐才有助于睡眠？',
    '我还没告诉你我叫什么，请介绍一下月社妃。',
    '我问你月社妃和琉璃的关系。', '今天请谈谈月社妃是哪本魔法之书的主人公。',
    '请给出原文依据。', '我对月社妃和夜子的关系很好奇。',
])
def test_external_or_mixed_tasks_are_not_suppressed(message):
    assert not local_context_only(message)
    assert RAGIntentDetector().needs_rag(message)[0] is True


@pytest.mark.parametrize('message', ['今天没有什么特别的事情。', '不用给我建议，我只是想说说今天的事。'])
def test_proven_local_task_bypasses_ml_loading(monkeypatch, message):
    from knowledge import intent_detector

    monkeypatch.setattr(intent_detector, '_ML_AVAILABLE', False)
    monkeypatch.setattr(intent_detector, '_load_ml_model', lambda: pytest.fail('No classifier needed'))
    assert intent_detector.needs_rag(message) == (
        False, '完整请求仅依赖个人上下文或为个人陈述，不需要RAG', None)


def test_unpunctuated_information_request_survives_negative_classifier(monkeypatch):
    from knowledge import intent_detector

    monkeypatch.setattr(intent_detector, '_ML_AVAILABLE', True)
    monkeypatch.setattr(intent_detector, '_ml_predict', lambda _: (False, 'negative', .99, None))
    assert intent_detector.needs_rag('不用给我建议我想知道月社妃和琉璃的关系')[0] is True


@pytest.mark.parametrize('message', ['我需要月社妃的原文。', '我想要关于黑洞的资料。',
                                   '我希望你谈谈数据库索引。'])
def test_desired_information_is_not_mislabeled_as_self_report(message):
    assert not local_context_only(message)


@pytest.mark.parametrize('message', [
    '我问你月社妃和琉璃的关系。', '我给你个任务：说说数据库索引的作用。',
    '今天请谈谈量子纠缠。', '我对量子力学的原理很好奇。',
    '我等你讲讲这段历史。', '现在麻烦推荐几本书。',
    '我喜欢历史并请推荐一些入门书。', '我喜欢烘焙也麻烦你推荐几本书。',
    '我喜欢摄影顺便列举一些构图方法。',
])
def test_owner_or_time_prefix_is_not_proof_of_a_declaration(message):
    assert not local_context_only(message)


@pytest.mark.asyncio
@pytest.mark.parametrize('message,external', [('今天没有什么特别的事情。', False),
    ('请记住，我的专业是材料学。', False), ('你还保存着我的专业吗？', False),
    ('请记住，我住在兰州。请解释季风形成的原因。', True),
    ('不用给我建议，我只是想说说今天的事。', False),
    ('不要分析，我只想聊聊自己的感受。', False),
    ('不用给我建议，请解释量子纠缠。', True),
    ('不用给我建议我想知道地球的年龄', True),
    ('我只有充分休息才开车。', False),
    ('更正一下，我只有收到确认才预约会议。', False),
    ('我只有设备检查通过才启动仪器，请解释仪器原理。', True),
    ('我的专业是统计学，现在在气象站工作。', False),
    ('我问你月社妃和琉璃的关系。', True), ('请给出原文依据。', True),
    ('我喜欢什么音乐？再介绍一下爵士乐。', True)])
async def test_production_route_preserves_task_ownership(monkeypatch, message, external):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    retrieved = []

    async def retrieve(*args):
        retrieved.append(args)
        return {'context_text': '外部资料', 'results': [], 'citations': [], 'abstained': False}

    async def model(**kwargs):
        return '正常回复'

    monkeypatch.setattr(intent_detector, '_ML_AVAILABLE', False)
    monkeypatch.setattr(intent_detector, '_load_ml_model', lambda: None)
    monkeypatch.setattr(generate, '_retrieve_rag_bundle', retrieve)
    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    await generate._generate_with_vllm(MessageRequest(message=message), None,
        runtime_config={'useKnowledgeBase': True}, model_generate=model)
    assert bool(retrieved) is external
