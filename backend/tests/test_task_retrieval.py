"""Independent candidate tasks require evidence slots, never invented facts."""

from copy import deepcopy

import pytest

from knowledge.task_retrieval import collect_task_candidates

QUERY = "Compare A B C D public rules without supplying any prices"
VIEWS = ("A public rules", "B public rules", "C public rules", "D public rules")


def row(name, score=0.8):
    return dict(
        id=name, title=name, content="Complete rule for " + name, knowledge_base_id=7, category="public", score=score
    )


class Corpus:
    def __init__(self):
        self.revision = 1
        self.calls = []
        self.data = {QUERY: [row("shared"), row("root")], **{q: [row("shared"), row(q.split()[0])] for q in VIEWS}}

    def retrieve(self, q, *, top_k, filters):
        self.calls.append((q, deepcopy(filters)))
        return deepcopy(self.data[q])

    def run(self, **overrides):
        args = dict(
            top_k=2,
            filters={"knowledge_base_id": 7},
            retrieve=self.retrieve,
            confidence=lambda rs: sum(x["score"] for x in rs) / len(rs) if rs else 0,
            stable_key=lambda r: r["id"],
            snapshot=lambda: self.revision,
        )
        args.update(overrides)
        return collect_task_candidates(QUERY, VIEWS, **args)


def baseline(c):
    p = c.run()
    assert {x["id"] for x in p.results} == {"shared", "root", "A", "B", "C", "D"}
    assert all(x["status"] == "candidates_retrieved" for x in p.coverage)
    c.calls.clear()
    return p


def test_four_independent_tasks_survive_a_two_candidate_local_limit():
    c = Corpus()
    p = baseline(c)
    assert len(p.results) == 6 <= 2 * 5
    shared = next(x for x in p.results if x["id"] == "shared")
    assert shared["retrieval_tasks"] == (0, 1, 2, 3, 4)
    assert all(x["semantic_coverage"] == "unverified" for x in p.coverage)


def test_duplicate_sources_are_not_new_votes_or_score_promotions():
    c = Corpus()
    baseline(c)
    before = deepcopy(c.data)
    c.data[VIEWS[0]][0]["score"] = 0.95
    p = c.run()
    shared = next(x for x in p.results if x["id"] == "shared")
    assert shared["score"] == 0.8 and "query_count" not in shared
    assert c.data[QUERY] == before[QUERY]


def test_missing_task_is_explicit_without_dropping_other_tasks():
    c = Corpus()
    baseline(c)
    c.data[VIEWS[2]] = []
    p = c.run()
    assert p.coverage[3]["status"] == "no_candidates"
    assert p.coverage[3]["semantic_coverage"] == "not_established"
    assert all(any(x["id"] == key for x in p.results) for key in ["A", "B", "D"])
    assert not any(3 in x["retrieval_tasks"] for x in p.results)


def test_weak_task_cannot_borrow_confidence_from_another_task():
    c = Corpus()
    baseline(c)
    c.data[VIEWS[3]] = [row("weak", 0.1)]
    p = c.run()
    assert p.coverage[4]["status"] == "low_confidence"
    assert not any(x["id"] == "weak" for x in p.results)


def test_filters_are_independent_copies_for_every_task():
    c = Corpus()
    baseline(c)
    filters = {"knowledge_base_id": [7]}

    def retrieve(q, **kwargs):
        assert kwargs["filters"] == filters
        kwargs["filters"]["knowledge_base_id"].append(8)
        return deepcopy(c.data[q])

    c.run(filters=filters, retrieve=retrieve)
    assert filters == {"knowledge_base_id": [7]}


def test_index_change_mid_bucket_rejects_partial_plan():
    c = Corpus()
    baseline(c)

    def retrieve(q, **kwargs):
        if q == VIEWS[1]:
            c.revision += 1
        return c.retrieve(q, **kwargs)

    with pytest.raises(RuntimeError, match="Index changed"):
        c.run(retrieve=retrieve)


def test_index_change_between_buckets_rejects_partial_plan():
    c = Corpus()
    baseline(c)
    reads = [0]

    def snapshot():
        reads[0] += 1
        return 2 if reads[0] >= 4 else 1

    with pytest.raises(RuntimeError, match="Index changed"):
        c.run(snapshot=snapshot)


def test_equal_id_with_changed_source_authority_rejects_plan():
    c = Corpus()
    baseline(c)
    c.data[VIEWS[1]][0]["content"] = "Different public source"
    with pytest.raises(RuntimeError, match="source authority"):
        c.run()


def test_provider_cannot_exceed_each_task_bound():
    c = Corpus()
    baseline(c)
    c.data[VIEWS[0]].append(row("extra"))
    with pytest.raises(RuntimeError, match="candidate bound"):
        c.run()


@pytest.mark.parametrize("limit", [False, 0, -1, "2"])
def test_invalid_per_task_limit_rejected_after_positive_baseline(limit):
    c = Corpus()
    baseline(c)
    with pytest.raises(ValueError):
        c.run(top_k=limit)


@pytest.mark.parametrize("threshold", [False, float("nan"), -1, 1.1])
def test_invalid_threshold_rejected_after_positive_baseline(threshold):
    c = Corpus()
    baseline(c)
    with pytest.raises(ValueError):
        c.run(threshold=threshold)
