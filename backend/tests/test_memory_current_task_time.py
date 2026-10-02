"""Complete task-local current requests keep unrelated date spans out of routing."""

from datetime import datetime, timezone

import pytest

from character.memory_query_time import current_lookup_time_text
from character.memory_service import _historical_query_window

NOW = datetime(2026, 10, 2, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "background",
    [
        "我在2017年参加过一次摄影培训。",
        "我的室友在2019年参加过一次摄影培训。",
        "我读过一本2020年出版的摄影教材。",
        "她2018年搬去了另一个城市；",
        "我在2020年三月参加了一次活动。",
    ],
)
def test_complete_current_task_has_its_own_time_focus(background):
    query = background + "请告诉我，我的现居地是什么？"
    assert current_lookup_time_text(query) == "请告诉我，我的现居地是什么"
    assert _historical_query_window(query, NOW) is None


@pytest.mark.parametrize(
    "task",
    [
        "我现在住在哪里？",
        "我目前住哪儿？",
        "我的当前工作单位是什么？",
        "我的当前名字和专业分别是什么？",
        "整理我的当前个人资料。",
    ],
)
def test_shared_closed_grammar_supports_current_fields_without_event_exceptions(task):
    query = "我在2017年参加过一次摄影培训。" + task
    assert current_lookup_time_text(query) is not None
    assert _historical_query_window(query, NOW) is None


@pytest.mark.parametrize(
    "query,year",
    [
        ("我在2017年参加过一次摄影培训。那时我的居住地是什么？", 2017),
        ("我在2017年参加过一次摄影培训。2026年9月我的居住地是什么？", 2026),
        ("我在2017年参加过一次摄影培训。我的住址是什么？", 2017),
        ("2017年。我的现居地是什么？", 2017),
        ("2017年我叫什么名字？我的现居地是什么？", 2017),
        ("请回忆2017年的事情。我的现居地是什么？", 2017),
        ("2017年我的当前住址是什么？", 2017),
    ],
)
def test_historical_frames_unknown_or_multiple_tasks_are_not_overridden(query, year):
    assert current_lookup_time_text(query) is None
    window = _historical_query_window(query, NOW)
    assert window is not None and window.start.year == year
