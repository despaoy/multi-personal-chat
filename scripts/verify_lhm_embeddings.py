"""Exercise current embedding code with existing lhm weights, offline and CPU only."""

import json
import shlex
from pathlib import Path

from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python

CHECK = r'''
import json, time
import numpy as np
import torch
from embedding import SentenceTransformerEmbeddingProvider
torch.set_num_threads(2)
provider = SentenceTransformerEmbeddingProvider(
    model_path='/home/boot/lhm/multipersonal-runtime/models/paraphrase-multilingual-MiniLM-L12-v2',
    batch_size=4,
)
memories = ['我喜欢喝乌龙茶，不喜欢咖啡。', '我下周三要参加研究生考试。',
            '我养了一只叫团子的猫。', '我每个周末都去游泳。']
queries = ['给我推荐一种饮料。', '最近有什么重要的考试安排？', '你记得我的宠物叫什么吗？', '我周末通常做什么运动？']
start = time.monotonic()
vectors = provider.embed_texts(memories)
assert str(provider._model.device) == 'cpu'
query_vectors = provider.embed_texts(queries)
scores = query_vectors @ vectors.T
ranking = scores.argmax(axis=1).tolist()
assert ranking == list(range(4)), ranking
assert np.isfinite(vectors).all()
assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
print(json.dumps({'device': str(provider._model.device), 'dimension': provider.dimension,
                  'fingerprint': provider.model_fingerprint, 'top1': ranking,
                  'model_id': provider.model_id, 'memories': memories, 'queries': queries,
                  'vectors': vectors.tolist(), 'query_vectors': query_vectors.tolist(),
                  'scores': scores.round(4).tolist(), 'seconds': round(time.monotonic()-start, 3)}))
'''


def main():
    workspace = Path(__file__).resolve().parents[1]
    with connect() as client:
        root = remote_python(client, "import tempfile\nprint(tempfile.mkdtemp(prefix='embedding-check-', dir='/home/boot/lhm/multipersonal-deploy'))\n").strip()
        assert root.startswith('/home/boot/lhm/multipersonal-deploy/embedding-check-')
        with client.open_sftp() as sftp:
            sftp.mkdir(root + '/infra')
            sftp.put(str(workspace / 'backend/knowledge/retrieval_core/embedding.py'), root + '/embedding.py')
            sftp.put(str(workspace / 'backend/infra/environment.py'), root + '/infra/environment.py')
            with sftp.file(root + '/verify.py', 'w') as handle:
                handle.write(CHECK.encode('utf-8'))
        env = {'HOME': root, 'TMPDIR': root, 'HF_HOME': root + '/hf', 'XDG_CACHE_HOME': root + '/cache',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
               'EMBEDDING_DEVICE': 'cpu',
               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'}
        command = 'env ' + ' '.join(k + '=' + shlex.quote(v) for k, v in env.items())
        command += ' /home/boot/lhm/multipersonal-runtime/venv/bin/python ' + shlex.quote(root + '/verify.py')
        _, stdout, stderr = client.exec_command(command, timeout=180)
        output, error = stdout.read().decode(), stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error)
        result = {'remote_root': root, 'result': json.loads(output)}
        summary = {k: v for k, v in result['result'].items() if k not in {'vectors', 'query_vectors'}}
        print(json.dumps({'remote_root': root, 'result': summary}, ensure_ascii=False, indent=2))
        target = workspace / '.codex-test-runtime-embeddings'
        target.mkdir(exist_ok=True)
        (target / 'latest-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
