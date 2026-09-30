"""Exercise the real HTTP transport/CLI against a synthetic loopback test service."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_persona_alignment_workflow import GENERATION, completed, judge_response, model, scene

from training.persona_alignment import main
from training.persona_sampling import chat_completion


@pytest.fixture
def endpoint():
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": payload})
            if payload["model"] == "http-error":
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b"secret-test-key")
                return
            content = (
                judge_response()
                if "response_format" in payload
                else ("归还小说" if payload["model"] == "local-candidate" else "归还笔记本")
            )
            body = {
                "choices": [
                    {
                        "finish_reason": "length" if payload["model"] == "truncated" else "stop",
                        "message": {"content": content},
                    }
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", captured
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def invoke(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["persona_alignment", *map(str, args)])
    main()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_cli_sample_blind_lock_report_judge_calibrate(endpoint, monkeypatch, tmp_path):
    url, captured = endpoint
    monkeypatch.setenv("TEST_PERSONA_KEY", "secret-test-key")
    models = [
        {**model(name), "base_url": url + "/v1", "api_key_env": "TEST_PERSONA_KEY"}
        for name in ("baseline", "candidate")
    ]
    scenes, config = tmp_path / "scenes.jsonl", tmp_path / "config.json"
    write(scenes, scene(split="validation"))
    write(config, {"models": models, "generation": GENERATION, "samples_per_model": 1})
    run, review, report = tmp_path / "run", tmp_path / "review", tmp_path / "report"
    invoke(monkeypatch, "sample", "--scenes", scenes, "--config", config, "--output-dir", run)
    assert len(captured) == 2
    assert all(item["path"] == "/v1/chat/completions" for item in captured)
    assert all(item["authorization"] == "Bearer secret-test-key" for item in captured)
    assert "secret-test-key" not in (run / "candidates.jsonl").read_text(encoding="utf-8")
    assert "secret-test-key" not in (run / "sampling_manifest.json").read_text(encoding="utf-8")
    invoke(
        monkeypatch,
        "blind",
        "--run",
        run,
        "--purpose",
        "evaluation",
        "--baseline",
        "baseline",
        "--candidate",
        "candidate",
        "--output-dir",
        review,
    )
    packet_path, key_path = review / "blind_review.json", review / "blind_key.json"
    packet, key = [json.loads(path.read_text(encoding="utf-8")) for path in (packet_path, key_path)]
    decisions, locked = tmp_path / "decisions.json", tmp_path / "locked.json"
    # Explicit synthetic human label only for this fixture; no project data is reviewed.
    write(decisions, completed(packet, key))
    invoke(monkeypatch, "lock", "--packet", packet_path, "--decisions", decisions, "--output", locked)
    review_args = ["--packet", packet_path, "--locked", locked, "--key", key_path]
    invoke(monkeypatch, "report", *review_args, "--output-dir", report)
    assert json.loads((report / "report.json").read_text())["counts"] == {"win": 1}
    judge_config, judgments = tmp_path / "judge.json", tmp_path / "judgments.jsonl"
    write(judge_config, {**model("judge"), "base_url": url})
    invoke(monkeypatch, "judge", "--run", run, "--model-config", judge_config, "--output", judgments)
    calibration = tmp_path / "calibration"
    invoke(monkeypatch, "calibrate", *review_args, "--judgments", judgments, "--output-dir", calibration)
    result = json.loads((calibration / "report.json").read_text())
    assert result["report"]["coverage"] == 1
    assert result["report"]["status"] == "measured_not_approved_for_ppo"
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "lock", "--packet", packet_path, "--decisions", decisions, "--output", locked)


def test_transport_rejects_truncation_and_redacts_error_body(endpoint):
    url, _ = endpoint
    for name, expected in (("truncated", "truncated"), ("http-error", "HTTP 429")):
        with pytest.raises((ValueError, RuntimeError), match=expected) as exc:
            chat_completion({**model(), "model": name, "base_url": url}, [{"role": "user", "content": "q"}], GENERATION)
        assert "secret-test-key" not in str(exc.value)


def test_invalid_model_config_rejected_before_writing_secrets(tmp_path, monkeypatch):
    scenes, config, output = tmp_path / "scenes.jsonl", tmp_path / "config.json", tmp_path / "run"
    write(scenes, scene())
    write(
        config,
        {"models": [{**model(), "api_key": "secret-test-key"}], "generation": GENERATION, "samples_per_model": 1},
    )
    with pytest.raises(ValueError, match="credentials"):
        invoke(monkeypatch, "sample", "--scenes", scenes, "--config", config, "--output-dir", output)
    assert not output.exists()
