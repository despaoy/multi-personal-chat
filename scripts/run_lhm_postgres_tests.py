"""Run current checkout against an isolated PostgreSQL cluster entirely inside lhm."""

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

MEMORY_DATABASES = {
    "STAGE31_FEEDBACK_PG_URL": ("stage3_stage31_feedback_guards", "test_message_completion_feedback.py"),
    "STAGE34_DELIVERY_PG_URL": ("stage3_stage34_delivery_guards", "test_delivery_memory.py"),
    "STAGE35_ADDRESS_PG_URL": ("stage3_stage35_address_guards", "test_turn_address_authority.py"),
    "STAGE38_RETENTION_PG_URL": ("stage3_stage38_retention_guards", "test_memory_selective_erasure.py"),
    "STAGE39_EVIDENCE_PG_URL": ("stage3_stage39_evidence_guards", "test_memory_retained_evidence.py"),
    "STAGE40_CLAUSE_PG_URL": ("stage3_stage40_clause_guards", "test_memory_coupled_erasure.py"),
    "STAGE42_FRAGMENTS_PG_URL": ("stage3_stage42_fragment_guards", "test_memory_cross_source_erasure.py"),
}

START = r'''
import json, os, pathlib, socket, subprocess, tempfile
base = pathlib.Path('/home/boot/lhm/multipersonal-deploy')
base.mkdir(exist_ok=True)
root = pathlib.Path(tempfile.mkdtemp(prefix='pg-foundation-', dir=base)).resolve()
assert root.is_relative_to(pathlib.Path('/home/boot/lhm'))
tools = pathlib.Path('/home/boot/lhm/multipersonal-deploy/tools/postgres-vector-14/usr/lib/postgresql/14/bin').resolve()
assert tools.is_relative_to(pathlib.Path('/home/boot/lhm'))
libraries = pathlib.Path('/home/boot/lhm/multipersonal-runtime/tools/pgsql/usr/lib/x86_64-linux-gnu').resolve()
assert libraries.is_relative_to(pathlib.Path('/home/boot/lhm'))
env = dict(os.environ, TMPDIR=str(root), HOME=str(root), LD_LIBRARY_PATH=str(libraries))
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
def run(args):
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
run([str(tools / 'initdb'), '-D', str(root / 'data'), '-A', 'trust', '-U', 'boot', '--no-locale', '-E', 'UTF8'])
run([str(tools / 'pg_ctl'), '-D', str(root / 'data'), '-l', str(root / 'postgres.log'),
     '-o', '-h 127.0.0.1 -p ' + str(port) + ' -k ' + str(root) + ' -c shared_buffers=16MB -c max_connections=40', '-w', 'start'])
print(json.dumps({'root': str(root), 'port': port, 'tools': str(tools), 'library': env.get('LD_LIBRARY_PATH', '')}))
'''


def remote_python(client, source):
    stdin, stdout, stderr = client.exec_command("python3 -", timeout=90)
    stdin.write(source)
    stdin.channel.shutdown_write()
    result, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", action="store_true", help="also run a real local Temporal worker against PostgreSQL")
    parser.add_argument("--pgvector", action="store_true", help="also test vector retrieval in the isolated PostgreSQL tools copy")
    parser.add_argument('--opensearch', action='store_true', help='test source stack using an already established OpenSearch tunnel')
    parser.add_argument("--real-embedding", action="store_true", help="generate real CPU vectors and replay them through pgvector/chat")
    parser.add_argument("--memory-boundaries", action="store_true", help="provision dedicated memory test databases; requires local port 25433")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if args.real_embedding:
        subprocess.run([sys.executable, str(workspace / 'scripts/verify_lhm_embeddings.py')], check=True)
    with connect() as ssh:
        instance = json.loads(remote_python(ssh, START))
        print("Isolated PostgreSQL:", instance["root"], flush=True)

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                channel = ssh.get_transport().open_channel(
                    "direct-tcpip", ("127.0.0.1", instance["port"]), self.request.getpeername())
                try:
                    while True:
                        readers, _, _ = select.select([self.request, channel], [], [], 1)
                        for reader in readers:
                            chunk = reader.recv(65536)
                            if not chunk:
                                return
                            (channel if reader is self.request else self.request).sendall(chunk)
                finally:
                    channel.close()

        class Tunnel(socketserver.ThreadingTCPServer):
            daemon_threads = True

        try:
            with Tunnel(("127.0.0.1", 25433 if args.memory_boundaries else 0), Handler) as tunnel:
                thread = threading.Thread(target=tunnel.serve_forever, daemon=True)
                thread.start()
                env = dict(os.environ, DATABASE_URL=f"postgresql+asyncpg://boot@127.0.0.1:{tunnel.server_address[1]}/postgres",
                           USE_POSTGRESQL="true", RUN_CHARACTER_PG_INTEGRATION="1")
                env["ALEMBIC_DATABASE_URL"] = env["DATABASE_URL"]
                tests = ["tests/test_character_chat_postgres.py", "tests/test_character_sources.py"]
                if args.opensearch:
                    if not env.get('TEST_OPENSEARCH_URL'):
                        raise RuntimeError('OpenSearch tunnel URL is required')
                    tests.extend(['tests/test_opensearch_runtime.py', 'tests/test_real_source_stack.py'])
                if args.pgvector or args.real_embedding:
                    env["RUN_PGVECTOR_INTEGRATION"] = "1"
                    tests.append("tests/test_pgvector_runtime.py")
                if args.real_embedding:
                    env['REAL_EMBEDDING_FIXTURE'] = str(workspace / '.codex-test-runtime-embeddings/latest-result.json')
                    tests.append('tests/test_real_embedding_chat.py')
                if args.temporal:
                    env["RUN_TEMPORAL_INTEGRATION"] = "1"
                    tests.append("tests/test_temporal_chat_integration.py")
                try:
                    if args.memory_boundaries:
                        # Only the cluster created above is eligible for provisioning.
                        source = "import os,subprocess\ninfo=" + repr(instance) + "\n"
                        source += "names=" + repr([name for name, _ in MEMORY_DATABASES.values()]) + "\n"
                        source += "for name in names:\n subprocess.run([info['tools']+'/createdb','-h','127.0.0.1','-p',str(info['port']),'-U','boot',name],env=dict(os.environ,LD_LIBRARY_PATH=info['library']),check=True)\n"
                        remote_python(ssh, source)
                        for variable, (name, test) in MEMORY_DATABASES.items():
                            env[variable] = f"postgresql+asyncpg://boot@127.0.0.1/{name}?port=25433"
                            tests.append("tests/" + test)
                        env["SOURCE_INHERIT_PG_URL"] = env["STAGE35_ADDRESS_PG_URL"]
                        tests.append("tests/test_memory_source_inheritance.py")
                    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=workspace / "backend", env=env, check=True)
                    result = subprocess.run([sys.executable, "-m", "pytest", *tests,
                                             "-q", "--basetemp", "../.pytest-tmp-character-pg"],
                                            cwd=workspace / "backend", env=env, check=False)
                finally:
                    tunnel.shutdown()
                    thread.join()
        finally:
            source = "import os,pathlib,subprocess\ninfo=" + repr(instance) + "\n"
            source += "root=pathlib.Path(info['root']).resolve()\nassert root.parent == pathlib.Path('/home/boot/lhm/multipersonal-deploy') and root.name.startswith('pg-foundation-')\n"
            source += "subprocess.run([info['tools']+'/pg_ctl','-D',str(root/'data'),'-m','fast','-w','stop'],env=dict(os.environ,LD_LIBRARY_PATH=info['library']),check=True)\n"
            print(remote_python(ssh, source), flush=True)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
