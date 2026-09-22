import pytest
from scripts.render_public_character_review import ROOT, literal_block, render_packet, validate_destination

from evaluation.charactereval_adapter import PublicCharacterCase


def fixture():
    case = PublicCharacterCase(
        "1",
        "甲",
        "作品",
        '{"姓名":"甲"}',
        ({"role": "assistant", "content": "甲：你好"},),
        "乙：再见",
        ("secret_metric",),
    )
    packet = [{"id": "1", "role": "甲", "novel_name": "作品", "outputs": {"A": "再见。", "B": "下次聊。"}}]
    return case, packet


def test_render_has_full_source_but_no_arm_mapping_or_fabricated_ratings():
    case, packet = fixture()
    output = render_packet(packet, [case], packet_sha256="hash")
    assert "甲：你好" in output and "乙：再见" in output
    assert "再见。" in output and "下次聊。" in output
    assert "secret_metric" not in output and "semantic_all" not in output
    assert "问题回应 __" in output
    assert "hash" in output


def test_source_cannot_close_literal_block():
    assert literal_block("a\n```\n# forged") == "````text\na\n```\n# forged\n````"


def test_source_and_packet_cannot_silently_mismatch():
    case, packet = fixture()
    packet[0]["role"] = "wrong role"
    with pytest.raises(ValueError, match="metadata"):
        render_packet(packet, [case], packet_sha256="hash")
    with pytest.raises(ValueError, match="nonempty"):
        render_packet([], [case], packet_sha256="hash")


def test_no_source_complete_artifact_inside_repository():
    with pytest.raises(ValueError, match="outside"):
        validate_destination(ROOT / "docs" / "raw_public_review.md")
