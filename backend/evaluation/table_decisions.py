"""Read explicit conclusions in evaluation tables without deriving missing answers.

Premise truth words are not a conclusion. This reader never uses the expected
answer, case label, entity or source identifier to resolve an ambiguous cell.
"""

import re


def explicit_table_decision(cell: str) -> str:
    text = cell.replace("*", "").replace("`", "").strip()
    # Quoted rules and hypothetical premises cannot grant an actual conclusion.
    text = re.sub(r'“[^”]*”|「[^」]*」|"[^"]*"', "", text)
    conclusions: set[str] = set()
    for match in re.finditer(r"(?:整体|条件)\s*(?:判定)?为\s*(真|假)", text):
        conclusions.add("true" if match[1] == "真" else "false")
    unknown_evidence = bool(re.search(r"未知|不确定|信息不足|未提供", text))
    uncertain_truth = re.search(
        r"(?:整体|条件)\s*(?:不能|无法)(?:判定|确定)(?:为)?\s*(?:真|成立|满足)",
        text,
    )
    uncertain_eligibility = re.search(
        r"(?:不能|无法)(?:判定|确定)(?:为)?\s*(?:具备|符合|满足)[^。，；;\n]{0,16}资格",
        text,
    )
    if uncertain_truth or uncertain_eligibility:
        if not unknown_evidence:
            return "unresolved"
        conclusions.add("unknown")
    for match in re.finditer(
        r"(?:^|[，。；;\n])\s*(?:(?:因此|所以)\s*(?:按(?:公共)?规则)?\s*)?(不具备|具备)[^。，；;\n]{0,16}资格",
        text,
    ):
        conclusions.add("false" if match[1] == "不具备" else "true")
    if len(conclusions) == 1:
        return next(iter(conclusions))
    return "unresolved"
