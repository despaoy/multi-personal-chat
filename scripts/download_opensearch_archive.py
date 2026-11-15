"""Download the official pinned archive locally for faster transfer into lhm."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

EXPECTED = 'a70e52a93ad66d9497dd734f5cc48eb4a8a8c4a92c47491a7ad0f1a898383d11c2948d4ba4159b2361d961bf221efa0fd02084e6389532df960b1ebe0c0a2099'
URL = 'https://artifacts.opensearch.org/releases/bundle/opensearch/3.9.0/opensearch-3.9.0-linux-x64.tar.gz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='resume only after confirming the previous downloader terminated')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / '.codex-test-runtime-opensearch'
    root.mkdir(exist_ok=True)
    archive = root / 'opensearch-3.9.0-linux-x64.tar.gz'
    partial = archive.with_suffix('.gz.part')
    if archive.exists() or (partial.exists() and not args.resume):
        raise RuntimeError('Archive or partial download exists; inspect its live process before retrying')
    started, size, digest = time.monotonic(), 0, hashlib.sha512()
    if partial.exists():
        with partial.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
                size += len(chunk)
    headers = {'Range': f'bytes={size}-'} if size else {}
    with httpx.stream('GET', URL, headers=headers, timeout=60) as response:
        response.raise_for_status()
        if size and (response.status_code != 206 or response.headers.get('content-range') != f'bytes {size}-1095236083/1095236084'):
            raise RuntimeError('Server did not honor the exact archive resume range')
        with partial.open('ab' if size else 'xb') as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size % (16 * 1024 * 1024) == 0:
                    status = {'bytes': size, 'seconds': round(time.monotonic() - started, 1)}
                    (root / 'local-download-status.json').write_text(json.dumps(status))
                    print(json.dumps(status), flush=True)
    if digest.hexdigest() != EXPECTED:
        raise RuntimeError('Official archive SHA512 mismatch')
    partial.rename(archive)
    print(json.dumps({'phase': 'verified', 'bytes': size, 'archive': str(archive)}), flush=True)


if __name__ == '__main__':
    main()
