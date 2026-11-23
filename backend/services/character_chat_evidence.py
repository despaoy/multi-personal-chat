"""Display snapshots come from admitted sources, never model-authored source labels."""

import re
from dataclasses import dataclass, field

from inference.citation_recovery import disclaims_requested_detail


@dataclass(frozen=True)
class CharacterChatCompletion:
    reply: str
    model: str
    evidence: dict = field(default_factory=dict)


def _only_expresses_uncertainty(question, reply):
    """Match standalone abstentions, including ones that repeat the question's topic."""
    if re.fullmatch(r"(?:我(?:也|可)?)?(?:不确定|说不准|不知道|不清楚|无法确认|不能确定)[。.!！]?", reply.strip()):
        return True
    answer = re.fullmatch(
        r"(?P<topic>[\w· ]+)[，,]\s*(?:(?:原作|原文|资料)没(?:有)?(?:交代|说明|提到)[，,]\s*)?"
        r"我(?:也|可)?(?:说不准|不知道|不清楚|无法确认|不能确定)[。.!！]?",
        reply.strip(),
    )
    if not answer:
        return False
    detail = re.search(r"是(?:什么|哪种|哪一种)(?P<field>[\w·]+)[？?]?$", question.strip())
    if not detail:
        return False
    topic = answer['topic'].replace('的', '')
    subject = question[:detail.start()].replace('的', '')
    for ending in (detail.group().rstrip('？?'), detail['field']):
        if topic.endswith(ending):
            echoed_subject = topic.removesuffix(ending)
            return bool(echoed_subject) and subject.endswith(echoed_subject)
    return False


def evidence_snapshot(result, *, question=""):
    retrieval = result.plan.retrieval
    documents = {}
    for item in retrieval.documents:
        identity = str(item.get("id") or item.get("chunk_id") or "")
        if not identity or identity in documents:
            raise ValueError("Retrieved source snapshot has missing or duplicate identity")
        documents[identity] = item
    sources = []
    for citation in result.response_citations:
        identity = str(citation.get("source_id") or citation.get("id") or "")
        if not identity:
            raise ValueError("Answer-used citation has no source identity")
        if identity not in documents:
            raise ValueError("Answer-used citation has no retrieved source snapshot")
        document = documents[identity]
        text = str(document.get("content") or document.get("text") or "")
        if question and disclaims_requested_detail(question, result.reply, (text,)):
            continue
        if question and _only_expresses_uncertainty(question, result.reply):
            # The same words can be an actual line of dialogue. Check the cited
            # document itself, so another retrieved document cannot vouch for it.
            utterance = result.reply.strip().rstrip("。.!！")
            if not re.search(r"(?<!\w)" + re.escape(utterance) + r"(?!\w)", text):
                continue
        sources.append(
            {
                "id": identity,
                "title": str(document.get("title") or document.get("original_title") or identity),
                "excerpt": text[:12000],
                "truncated": len(text) > 12000,
                "line_start": citation.get("line_start"),
                "line_end": citation.get("line_end"),
            }
        )
    return {
        "sources": sources,
        "retrieval_status": retrieval.status,
        "response_mode": result.response_mode,
        "reason": retrieval.reason,
    }
