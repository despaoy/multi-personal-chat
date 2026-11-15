"""Transfer the locally verified archive; use a fresh lhm installation directory."""

import json
from pathlib import Path

from prepare_lhm_opensearch import WORKER
from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python


def main():
    directory = Path(__file__).resolve().parents[1] / '.codex-test-runtime-opensearch'
    archive = directory / 'opensearch-3.9.0-linux-x64.tar.gz'
    if not archive.is_file():
        raise RuntimeError('Local verified download is not ready')
    with connect() as ssh:
        root = remote_python(ssh, "import tempfile\nprint(tempfile.mkdtemp(prefix='opensearch-local-',dir='/home/boot/lhm/multipersonal-deploy/tools'))\n").strip()
        assert root.startswith('/home/boot/lhm/multipersonal-deploy/tools/opensearch-local-')
        with ssh.open_sftp() as sftp:
            print('Transferring verified archive to ' + root, flush=True)
            sftp.put(str(archive), root + '/opensearch.tar.gz')
            with sftp.file(root + '/prepare.py', 'w') as handle:
                handle.write(WORKER.encode())
        source = 'import pathlib,subprocess,json\nroot=pathlib.Path(' + repr(root) + ')\n'
        source += "with (root/'prepare.log').open('wb') as log:\n    process=subprocess.Popen(['python3',str(root/'prepare.py')],cwd=root,stdout=log,stderr=log,start_new_session=True)\n"
        source += "print(json.dumps({'root':str(root),'pid':process.pid}))\n"
        state = json.loads(remote_python(ssh, source))
        (directory / 'local-transfer.json').write_text(json.dumps(state), encoding='utf-8')
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
