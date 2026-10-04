"""Select literal source references without asking reviewers to rewrite quotes."""

import json

from knowledge.public_object_scope import _unique, validate_object_scopes

SPAN_INSTRUCTION = """object_scopes只根据完整问题绑定查询对象；identity_review只确认完整授权正文中的单跳名称同一性。
source_span_catalog是程序从当前完整来源抽取的逐字片段，只提供可选择位置，不代表任务相关或事实充分。完整sources仍是事实审核依据。
审核每个来源是否支持public_task_ids中的任务、对象及共同限定；没有依据时task_ids为空，不能借另一业务规则或私人偏好补齐。
不要生成、复制或改写source_quote。只选source_span_id，程序将提取它对应的真实原文。片段必须属于当前source_id并包含该对象完整query_text，或包含已核验identity_review中对应别名。
别名规则的object_evidence必须同时使用对应identity_binding_id，登记来源与规则须同知识库；不得把规则中的别名替换为问题正式名。identity_only登记不提供业务事实，task_ids为空。
每个required_source_ids恰好一次，仅输出严格JSON：
{"decisions":[{"source_id":"允许来源","task_ids":["允许任务"],"object_evidence":[{"object_id":"绑定对象","source_span_id":"当前来源可选位置"}]}]}。
仅别名object_evidence可加identity_binding_id；不输出source_quote、解释、额外键或代码块。source_span_id不是source_id或object_id，须取本轮实际目录。未解决对象及不存在的名称关系保持未知。"""


class SourceSpanCapacityError(ValueError):
    """A complete source-reference catalogue exceeds this profile."""


def build_source_spans(payload, scopes):
    validate_object_scopes(scopes, payload["query"], payload["public_task_ids"])
    from knowledge.public_identity_dependencies import validate_identity_receipt

    identity = validate_identity_receipt(payload, scopes)
    names = [obj["query_text"] for obj in scopes["objects"]]
    catalogue = []
    for source in payload["sources"]:
        aliases = [
            row["alias"]
            for row in identity["bindings"]
            if type(source.get("knowledge_base_id")) is int
            and source["knowledge_base_id"] == row["knowledge_base_id"]
            and source["source_id"] != row["source_id"]
        ]
        parts = [source.get("original_body"), *[chunk["content"] for chunk in source["indexed_chunks"]]]
        quotes = set()
        for part in parts:
            if not isinstance(part, str) or not part.strip():
                continue
            for name in dict.fromkeys([*names, *aliases]):
                offset = part.find(name)
                if offset < 0:
                    continue
                start = max(0, offset - 64)
                quote = part[start : start + 512]
                if name not in quote or quote in quotes:
                    continue
                if len(catalogue) >= 256:
                    raise SourceSpanCapacityError("Complete source reference catalogue exceeds capacity")
                catalogue.append(
                    dict(span_id=f"source-span:{len(catalogue)}", source_id=source["source_id"], source_quote=quote)
                )
                quotes.add(quote)
    return catalogue


def expand_span_decisions(raw, payload, scopes):
    catalogue = build_source_spans(payload, scopes)
    if payload.get("source_span_catalog") != catalogue:
        raise ValueError("Source reference catalogue changed the authorized original input")
    references = {row["span_id"]: row for row in catalogue}
    value = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(value, dict) or set(value) != {"decisions"} or not isinstance(value["decisions"], list):
        raise ValueError("Invalid source reference envelope")
    rows = []
    for row in value["decisions"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"source_id", "task_ids", "object_evidence"}
            or not isinstance(row["object_evidence"], list)
        ):
            raise ValueError("Invalid source reference decision")
        proofs = []
        for proof in row["object_evidence"]:
            if not isinstance(proof, dict) or set(proof) not in (
                {"object_id", "source_span_id"},
                {"object_id", "source_span_id", "identity_binding_id"},
            ):
                raise ValueError("Object evidence must select an actual source reference")
            sid = proof["source_span_id"]
            if not isinstance(sid, str) or sid not in references or references[sid]["source_id"] != row["source_id"]:
                raise ValueError("Unknown or cross-source reference")
            expanded = dict(object_id=proof["object_id"], source_quote=references[sid]["source_quote"])
            if "identity_binding_id" in proof:
                expanded["identity_binding_id"] = proof["identity_binding_id"]
            proofs.append(expanded)
        rows.append(dict(source_id=row["source_id"], task_ids=row["task_ids"], object_evidence=proofs))
    return json.dumps(dict(decisions=rows), ensure_ascii=False)
