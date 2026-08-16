from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ny = ZoneInfo("America/New_York")
wall_time = datetime(2024, 11, 3, 1, 30)
earlier = wall_time.replace(tzinfo=ny, fold=0)
later = wall_time.replace(tzinfo=ny, fold=1)
earlier_utc = earlier.astimezone(timezone.utc)
later_utc = later.astimezone(timezone.utc)
assert earlier.utcoffset().total_seconds() == -4 * 3600
assert later.utcoffset().total_seconds() == -5 * 3600
assert earlier_utc == datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc)
assert later_utc == datetime(2024, 11, 3, 6, 30, tzinfo=timezone.utc)
assert later_utc - earlier_utc == timedelta(hours=1)
print("technical verification passed")
