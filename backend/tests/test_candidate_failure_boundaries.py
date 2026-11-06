"""Complete frozen/enriched fixtures through actual candidate transport and CLI callers."""

import importlib.util
import io
import json
import urllib.error
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import knowledge.game_rag.ollama_client as transport
from knowledge.game_rag.ollama_client import OllamaCandidateClient


@pytest.fixture(params=["knowledge", "metadata"])
def candidate_case(request, tmp_path):
    kind = request.param
    if kind == "knowledge":
        import test_game_rag_knowledge_candidate as data

        from knowledge.game_rag.knowledge_candidate import (
            create_knowledge_run,
            generate_knowledge_candidates,
            load_knowledge_run,
            save_knowledge_run,
        )

        scenes = [
            data._enriched_scene(
                f"scene_native_{i}",
                i * 10 + 1,
                i * 10 + 10,
                text="\n".join(
                    [
                        "[琉璃] 我是从本岛转学回到小岛的少年。",
                        "这里记录的是我当前的身份。",
                        *[f"补充叙述{n}。" for n in range(8)],
                    ]
                ),
                present=["琉璃"],
                mentioned=["琉璃"],
            )
            for i in range(2)
        ]
        bundle = data._load_bundle(tmp_path / "source", scenes)
        state = create_knowledge_run(bundle, model_id="synthetic-candidate-model")
        payloads = [
            data._payload_json(
                s.id, facts=[data._fact_item(line_start=s.source.line_start, line_end=s.source.line_start + 1)]
            )
            for s in scenes
        ]
        generate = partial(generate_knowledge_candidates, alias_config=data.ALIASES)
        load, save = load_knowledge_run, save_knowledge_run
    else:
        import test_game_rag_scene_metadata_candidate as data

        from knowledge.game_rag.scene_metadata_candidate import (
            create_candidate_run,
            generate_scene_candidates,
            load_candidate_run,
            save_candidate_run,
        )

        scenes = [
            s.model_copy(
                update={
                    "text": "\n".join(
                        [
                            "[妃] 我正在小岛上记述眼前的现实事件。",
                            "[妃] 这是我现在的第一人称叙述。",
                            *[f"当前场景叙述{n}。" for n in range(s.source.line_end - s.source.line_start - 1)],
                        ]
                    ),
                    "present_characters": ["妃"],
                    "mentioned_characters": ["妃"],
                }
            )
            for s in data._default_scenes()
        ]
        bundle = data._load_bundle(tmp_path / "source", scenes)
        state = create_candidate_run(bundle, model_id="synthetic-candidate-model")
        payloads = [data._candidate_json(s.id, s.source.line_start, mentioned=["妃"], present=["妃"]) for s in scenes]
        generate = generate_scene_candidates
        load, save = load_candidate_run, save_candidate_run
    path = tmp_path / "candidate-run.json"
    save(path, state)
    client = OllamaCandidateClient(
        base_url="http://synthetic-candidate.test",
        model=state.model_id,
        temperature=0.2,
        top_p=0.9,
        num_predict=3072,
        num_ctx=16384,
        timeout_s=0.5,
    )
    return SimpleNamespace(
        kind=kind,
        data=data,
        bundle=bundle,
        state=state,
        payloads=payloads,
        generate=generate,
        load=load,
        path=path,
        client=client,
    )


def response(content):
    return io.BytesIO(
        json.dumps(
            {
                "message": {"role": "assistant", "content": content},
                "done": True,
                "prompt_eval_count": 30,
                "eval_count": 10,
            }
        ).encode()
    )


def test_permanent_transport_fault_preserves_checkpoint_and_resume(candidate_case, monkeypatch):
    case = candidate_case
    requests = []

    def interrupted(request, *, timeout):
        body = json.loads(request.data)
        assert body["model"] == case.state.model_id and timeout == 0.5
        assert any(scene.id in body["messages"][0]["content"] for scene in case.bundle.scenes)
        requests.append(body)
        if len(requests) >= 2:
            raise urllib.error.HTTPError(
                request.full_url, 401, "synthetic unauthorized", None, io.BytesIO(b"synthetic")
            )
        return response(case.payloads[0])

    monkeypatch.setattr(transport.urllib.request, "urlopen", interrupted)
    with pytest.raises(RuntimeError, match="ollama http status 401") as error:
        case.generate(case.bundle, case.state, case.client, state_path=case.path)
    assert isinstance(error.value.__cause__, urllib.error.HTTPError)
    assert error.value.__cause__.code == 401
    error.value.__cause__.close()
    assert case.client.stats["calls"] == 2 and case.client.stats["failures"] == 1
    assert len(requests) == 2
    checkpoint = case.load(case.path)
    assert checkpoint.scene_states[0].status.value == "success"
    assert all(item.status.value == "pending" for item in checkpoint.scene_states[1:])
    completed = checkpoint.scene_states[0].model_dump(mode="json")

    remaining = iter(case.payloads[1:])
    recovered_requests = []

    def recovered(request, *, timeout):
        recovered_requests.append(json.loads(request.data))
        return response(next(remaining))

    monkeypatch.setattr(transport.urllib.request, "urlopen", recovered)
    result = case.generate(case.bundle, checkpoint, case.client, state_path=case.path)
    assert result.succeeded_scene_ids == [s.id for s in case.bundle.scenes[1:]]
    assert len(recovered_requests) == len(case.bundle.scenes) - 1
    final = case.load(case.path)
    assert all(item.status.value == "success" for item in final.scene_states)
    assert final.scene_states[0].model_dump(mode="json") == completed


@pytest.mark.parametrize(
    "candidate_case,entry",
    [("knowledge", "shared"), ("metadata", "smoke"), ("metadata", "run")],
    indirect=["candidate_case"],
)
def test_candidate_cli_reports_failed_run_after_saving_truth(candidate_case, entry, monkeypatch, tmp_path):
    case = candidate_case
    filename = "run_knowledge_candidates.py" if case.kind == "knowledge" else "run_scene_metadata_candidates.py"
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location("synthetic_candidate_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    out = case.path.parent
    manifest_path = out / "run-manifest.json"
    monkeypatch.setattr(runner, "OUT_DIR", out)
    monkeypatch.setattr(runner, "STATE_PATH", case.path)
    monkeypatch.setattr(runner, "MANIFEST_OUT_PATH", manifest_path)
    monkeypatch.setattr(runner, "MODEL_ID", case.state.model_id)
    monkeypatch.setattr(runner, "_model_client", lambda: case.client)
    monkeypatch.setattr(
        transport.urllib.request, "urlopen", lambda request, timeout: response("invalid synthetic JSON")
    )
    if case.kind == "knowledge":
        monkeypatch.setattr(runner, "_load_inputs", lambda: (case.bundle, case.data.ALIASES))
        call = partial(runner._run_phase, None, "isolated")
    else:
        monkeypatch.setattr(runner, "GENERATION_PARAMS", case.state.generation_params)
        monkeypatch.setattr(runner, "SMOKE_SCENE_IDS", [s.id for s in case.bundle.scenes])
        monkeypatch.setattr(runner, "CALL_STATS_PATH", out / "call-stats.json")
        monkeypatch.setattr(runner, "load_bundle_with_gate", lambda: case.bundle)
        call = runner.phase_smoke if entry == "smoke" else runner.phase_run
    with pytest.raises(SystemExit) as error:
        call()
    assert error.value.code == 1
    persisted = case.load(case.path)
    assert all(item.status.value == "failed" and item.attempts == 1 for item in persisted.scene_states)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scene_status_counts"]["failed"] == len(case.bundle.scenes)
    assert case.client.stats["calls"] == len(case.bundle.scenes)
