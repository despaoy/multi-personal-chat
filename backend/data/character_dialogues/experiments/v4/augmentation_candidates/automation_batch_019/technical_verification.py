from __future__ import annotations


def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags


first = add_tag("red")
second = add_tag("blue")
assert first == ["red"]
assert second == ["blue"]
assert first is not second

existing = ["green"]
returned = add_tag("yellow", existing)
assert existing == ["green", "yellow"]
assert returned is existing

_MISSING = object()


def add_tag_strict(tag, tags=_MISSING):
    if tags is _MISSING:
        tags = []
    elif tags is None:
        raise TypeError("tags must be a list, not None")
    tags.append(tag)
    return tags


assert add_tag_strict("red") == ["red"]
items = []
assert add_tag_strict("blue", items) is items
assert items == ["blue"]
try:
    add_tag_strict("bad", None)
except TypeError as exc:
    assert str(exc) == "tags must be a list, not None"
else:
    raise AssertionError("expected TypeError")

print("technical verification passed")
