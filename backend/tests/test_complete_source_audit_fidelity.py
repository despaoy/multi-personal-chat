import pytest

from evaluation.complete_source_role_audit import original_source_answer_fidelity

BODIES = ["HX01：数值21，限制：禁用。", "HX02：数值22，限制：复核后使用。"]


@pytest.mark.parametrize("separator", ["\n", "\n\n", "  \n"])
def test_exact_bodies_allow_only_markdown_line_separators(separator):
    result = original_source_answer_fidelity(separator.join(BODIES), BODIES)
    assert result["exact_bodies_in_order_no_content_additions"]
    assert result["byte_identical_to_blank_line_join"] == (separator == "\n\n")
    assert result["markdown_hard_line_breaks"] == (separator == "  \n")


@pytest.mark.parametrize(
    "answer",
    [
        "\n".join(reversed(BODIES)),
        BODIES[0],
        "\n".join(BODIES).replace("数值21", "数值23"),
        "\n".join(BODIES).replace("限制：禁用。", ""),
        "以下是记录：\n" + "\n".join(BODIES),
        "\n".join(BODIES) + "\n补充说明",
        " \n".join(BODIES),
        "\n".join(BODIES) + " ",
    ],
)
def test_source_content_or_unapproved_whitespace_changes_fail(answer):
    assert not original_source_answer_fidelity(answer, BODIES)["exact_bodies_in_order_no_content_additions"]


def test_source_owned_whitespace_is_never_stripped():
    bodies = ["  原话一 ", "原话二"]
    assert original_source_answer_fidelity("\n".join(bodies), bodies)["exact_bodies_in_order_no_content_additions"]
    assert not original_source_answer_fidelity("原话一\n原话二", bodies)["exact_bodies_in_order_no_content_additions"]


def test_empty_expected_sources_are_not_a_success():
    assert not original_source_answer_fidelity("", [])["exact_bodies_in_order_no_content_additions"]
