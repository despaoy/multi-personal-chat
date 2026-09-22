"""Download a public pinned model with bounded HTTP ranges and SHA-256 checking.

Useful when a large single HTTP stream stalls. No credentials or remote code
are used. Existing destinations/staging files are never overwritten. The final
path is published only after size and checksum verification.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def fetch_range(url, start, end):
    parts = urlsplit(url)
    query = urlencode([*parse_qsl(parts.query), ("download_range", str(start))])
    ranged_url = urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
    request = Request(ranged_url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "verified-model-download/1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                    f"bytes {start}-{end}/"
                ):
                    raise ValueError("server did not honor exact byte range")
                data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("incorrect range response length")
                return start, data
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (
        urlsplit(args.url).scheme != "https"
        or args.size <= 0
        or not 1 <= args.chunk_mib <= 32
        or not 1 <= args.workers <= 8
    ):
        parser.error("HTTPS, positive size and bounded chunk/worker settings required")
    if len(args.sha256) != 64 or any(char not in "0123456789abcdef" for char in args.sha256):
        parser.error("expected a lowercase SHA-256 digest")
    staging = args.output.with_suffix(args.output.suffix + ".verified-download.part")
    if args.output.exists() or staging.exists():
        parser.error("destination or staging file already exists; inspect it before retrying")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mib * 1024 * 1024
    with staging.open("xb+") as destination, ThreadPoolExecutor(max_workers=args.workers) as pool:
        destination.truncate(args.size)
        ranges = iter(range(0, args.size, chunk))
        jobs = set()

        def submit_next():
            start = next(ranges, None)
            if start is not None:
                jobs.add(pool.submit(fetch_range, args.url, start, min(args.size - 1, start + chunk - 1)))

        for _ in range(args.workers):
            submit_next()
        downloaded = 0
        while jobs:
            completed, jobs = wait(jobs, return_when=FIRST_COMPLETED)
            for future in completed:
                start, data = future.result()
                destination.seek(start)
                destination.write(data)
                downloaded += len(data)
                if downloaded == args.size or downloaded % (chunk * 8) < chunk:
                    print(f"Downloaded {downloaded}/{args.size} bytes", flush=True)
                submit_next()
        destination.flush()
        os.fsync(destination.fileno())
    digest = hashlib.sha256()
    with staging.open("rb") as source:
        while data := source.read(8 * 1024 * 1024):
            digest.update(data)
    if digest.hexdigest() != args.sha256:
        raise ValueError("SHA-256 mismatch; staging file retained for inspection, final path not published")
    # rename refuses an existing destination on the supported Windows host.
    if args.output.exists():
        raise FileExistsError(args.output)
    staging.rename(args.output)
    print(f"Verified {args.sha256}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
