from csv import DictReader
from decimal import Decimal, InvalidOperation
from io import StringIO


def total_by_item(text):
    reader = DictReader(StringIO(text, newline=""))
    required = {"item", "amount"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    totals = {}
    for line_no, row in enumerate(reader, start=2):
        item = (row["item"] or "").strip()
        raw_amount = (row["amount"] or "").strip()
        if not item and not raw_amount:
            continue
        if not item:
            raise ValueError(f"line {line_no}: empty item")
        try:
            amount = Decimal(raw_amount)
        except InvalidOperation as exc:
            raise ValueError(f"line {line_no}: invalid amount {raw_amount!r}") from exc
        totals[item] = totals.get(item, Decimal("0")) + amount
    return totals


sample = (
    'item,amount\r\n'
    '"tea, black",12.50\r\n'
    'cake,7.25\r\n'
    '"tea, black",2.00\r\n'
)
assert total_by_item(sample) == {
    "tea, black": Decimal("14.50"),
    "cake": Decimal("7.25"),
}
try:
    total_by_item("item,amount\ncoffee,nope\n")
except ValueError as exc:
    assert str(exc) == "line 2: invalid amount 'nope'"
else:
    raise AssertionError("invalid amount was accepted")

print("CSV technical verification passed")
