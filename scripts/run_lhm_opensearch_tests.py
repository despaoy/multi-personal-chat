"""Tunnel to the recorded isolated node and run the real search contract test."""

import argparse
import json
import os
import select
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

from probe_lhm_runtime import connect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--postgres', action='store_true')
    parser.add_argument('--real-embedding', action='store_true')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if args.real_embedding:
        subprocess.run([sys.executable, str(workspace / 'scripts/verify_lhm_embeddings.py')], check=True)
    instance = json.loads((workspace / '.codex-test-runtime-opensearch/instance.json').read_text())
    with connect() as ssh:
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                channel = ssh.get_transport().open_channel('direct-tcpip', ('127.0.0.1', instance['port']), self.request.getpeername())
                try:
                    while True:
                        readers, _, _ = select.select([self.request, channel], [], [], 1)
                        for reader in readers:
                            data = reader.recv(65536)
                            if not data:
                                return
                            (channel if reader is self.request else self.request).sendall(data)
                finally:
                    channel.close()

        class Tunnel(socketserver.ThreadingTCPServer):
            daemon_threads = True

        with Tunnel(('127.0.0.1', 0), Handler) as tunnel:
            thread = threading.Thread(target=tunnel.serve_forever, daemon=True)
            thread.start()
            try:
                env = dict(os.environ, TEST_OPENSEARCH_URL=f'http://127.0.0.1:{tunnel.server_address[1]}')
                if args.real_embedding:
                    env['REAL_EMBEDDING_FIXTURE'] = str(workspace / '.codex-test-runtime-embeddings/latest-result.json')
                command = ([sys.executable, str(workspace / 'scripts/run_lhm_postgres_tests.py'), '--opensearch'] if args.postgres else
                           [sys.executable, '-m', 'pytest', 'tests/test_opensearch_runtime.py', 'tests/test_real_source_stack.py', '-q',
                            '--basetemp', '../.pytest-tmp-opensearch-real'])
                result = subprocess.run(command, cwd=workspace / 'backend', env=env, check=False)
                return result.returncode
            finally:
                tunnel.shutdown()
                thread.join()


if __name__ == '__main__':
    raise SystemExit(main())
