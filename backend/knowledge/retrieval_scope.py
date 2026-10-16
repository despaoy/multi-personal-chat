"""Apply source-blind task scopes to structured ownership, never to permissions."""

from dataclasses import dataclass

from knowledge.entity_scope import in_identity_scope
from knowledge.relation_scope import in_relation_scope


def validate_scope_policy(policy):
    if (
        not isinstance(policy, dict)
        or set(policy) != {"allow_independent_identity", "include_unresolved"}
        or any(type(value) is not bool for value in policy.values())
    ):
        raise ValueError("Invalid declared retrieval scope policy")
    return policy


@dataclass(frozen=True)
class QueryScope:
    domain_id: str
    query: str
    rules: tuple[tuple[str | int, str, tuple[str, ...]], ...]
    include_unresolved: bool = False

    def allows(self, document):
        if self.include_unresolved:
            return True
        for _, mode, names in self.rules:
            if mode == "open":
                return True
            if mode in {"identity", "independent_identity"}:
                allowed = in_identity_scope(document, names[0])
            elif mode == "direct_relation":
                allowed = in_relation_scope(document, frozenset(names))
            else:
                raise ValueError("Unknown semantic retrieval scope")
            if allowed:
                return True
        return False

    @property
    def identity_subject(self):
        return (
            self.rules[0][2][0]
            if not self.include_unresolved
            and len(self.rules) == 1
            and self.rules[0][1] in {"identity", "independent_identity"}
            else ""
        )

    @property
    def independent_identity(self):
        return bool(self.identity_subject) and self.rules[0][1] == "independent_identity"

    def describe(self, excluded_candidates):
        return dict(
            status="task_bound",
            rules=[dict(task_id=identity, mode=mode, objects=list(names)) for identity, mode, names in self.rules],
            include_unresolved=self.include_unresolved,
            excluded_candidates=excluded_candidates,
        )


def scope_for_plan(plan, query):
    """Project each validated task onto this source; independent tasks form a union."""
    payload = plan["binding"]["input"]
    if "scope_policy" not in payload:
        raise ValueError("Character retrieval requires a semantic task scope")
    policy = validate_scope_policy(payload["scope_policy"])
    if payload["query"] not in query:
        raise ValueError("Semantic retrieval scope discarded the complete question")
    objects = {obj["object_id"]: obj for obj in plan["objects"]}
    rules = []
    for task in plan["tasks"]:
        if task["authorities"] and "curated_character" not in task["authorities"]:
            continue
        owned = [
            objects[identity]
            for identity in task["object_ids"]
            if objects[identity]["authority"] == "curated_character"
        ]
        mode = task["retrieval_scope"]
        if not owned or len(owned) != len(task["object_ids"]):
            # A source projection cannot narrow a task whose other objects it
            # does not own. It also cannot invent an unresolved object's name.
            mode = "open"
        names = tuple(obj["canonical_entity"] or obj["query_text"] for obj in owned)
        if mode != "open" and any(obj["canonical_entity"] is None for obj in owned):
            raise ValueError("Closed character scope requires registered character names")
        rules.append((task["task_id"], mode, names))
    if not rules and not policy["include_unresolved"]:
        raise ValueError("No character or unresolved task belongs to this source")
    return QueryScope(plan["domain_id"], query, tuple(rules), policy["include_unresolved"])
