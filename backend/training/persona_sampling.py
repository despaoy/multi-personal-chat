"""Reviewed scenes -> real model candidates, with immutable context bindings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request

from inference.prompt_policy import build_grounded_user_message
from training.preference_validation import fingerprint


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def validate_scene(scene):
    for key in ("id", "persona", "source_group", "persona_profile", "relationship", "situation", "user_message"):
        require_text(scene.get(key), key)
    if scene.get("split") not in {"train", "validation", "test"} or scene.get("review_status") != "approved":
        raise ValueError("scene needs an explicit partition and approved review")
    sources = scene.get("source_ids")
    if not isinstance(sources, list) or not sources or any(not isinstance(x, str) or not x.strip() for x in sources):
        raise ValueError("scene source_ids are required")
    evidence = scene.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list (empty for unknown facts)")
    seen = set()
    for item in evidence:
        for key in ("id", "source_id", "content"):
            require_text(item.get(key), f"evidence.{key}")
        if item["id"] in seen or item["source_id"] not in sources:
            raise ValueError("duplicate evidence id or undeclared evidence source")
        seen.add(item["id"])
    history = scene.get("history", [])
    if not isinstance(history, list) or len(history) % 2:
        raise ValueError("history must contain complete user/assistant turns")
    for index, item in enumerate(history):
        if not isinstance(item, dict) or item.get("role") != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("history must alternate user/assistant")
        require_text(item.get("content"), "history.content")


def scene_messages(scene):
    validate_scene(scene)
    # Only reviewed profile/relationship/state become application instructions.
    system = (
        f"{scene['persona_profile']}\n\n已审核关系状态：{scene['relationship']}\n"
        f"当前情境：{scene['situation']}\n根据已知情境回应；没有依据时不要补造经历或关系。"
    )
    evidence = "\n".join(f"[{item['id']}] {item['content']}" for item in scene["evidence"])
    user = build_grounded_user_message(scene["user_message"], evidence, max_chars=max(1, len(evidence)))
    return [{"role": "system", "content": system}, *scene.get("history", []), {"role": "user", "content": user}]


def validate_scenes(scenes):
    if not scenes:
        raise ValueError("scenes cannot be empty")
    ids, owners = set(), {}
    for scene in scenes:
        validate_scene(scene)
        if scene["id"] in ids:
            raise ValueError("duplicate scene id")
        ids.add(scene["id"])
        keys = [
            ("group", scene["source_group"]),
            ("prompt", fingerprint(scene_messages(scene))),
            *(("source", value) for value in scene["source_ids"]),
            *(("evidence", fingerprint(item["content"])) for item in scene["evidence"]),
        ]
        for key in keys:
            if owners.setdefault(key, scene["split"]) != scene["split"]:
                raise ValueError("cross-split scene/evidence leakage")


def validate_model(model):
    for key in ("name", "model", "revision", "base_url"):
        require_text(model.get(key), f"model.{key}")
    url = urllib.parse.urlsplit(model["base_url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("base_url must be HTTP(S) without credentials, query or fragment")
    allowed = {"name", "model", "revision", "base_url", "api_key_env"}
    if set(model) - allowed:
        raise ValueError("unknown model field; credentials must be supplied through api_key_env")
    if model.get("api_key_env") is not None:
        require_text(model["api_key_env"], "api_key_env")


def validate_generation(generation):
    if set(generation) != {"temperature", "top_p", "max_tokens", "seed"}:
        raise ValueError("generation must specify temperature, top_p, max_tokens and seed")
    for key in ("temperature", "top_p"):
        value = generation[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("generation sampling values must be finite numbers")
    if not 0 <= generation["temperature"] <= 2 or not 0 < generation["top_p"] <= 1:
        raise ValueError("invalid sampling range")
    if type(generation["max_tokens"]) is not int or generation["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if type(generation["seed"]) is not int or generation["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")


def chat_completion(model, messages, generation, *, json_mode=False, timeout=120):
    """OpenAI-compatible transport; never echo an API error body or credential."""
    validate_model(model)
    validate_generation(generation)
    headers = {"Content-Type": "application/json"}
    if model.get("api_key_env"):
        key = os.environ.get(model["api_key_env"])
        if not key:
            raise ValueError("configured API key environment variable is unset")
        headers["Authorization"] = "Bearer " + key
    payload = {"model": model["model"], "messages": messages, **generation}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    base = model["base_url"].rstrip("/")
    endpoint = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"model endpoint returned HTTP {exc.code}") from None
    except (OSError, ValueError):
        raise RuntimeError("model endpoint failed or returned invalid JSON") from None
    try:
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("model response was truncated, filtered, or incomplete")
        return require_text(choice["message"]["content"], "model response")
    except (KeyError, IndexError, TypeError):
        raise ValueError("invalid model response schema") from None


def sampling_contract(scenes, models, generation, samples_per_model):
    validate_scenes(scenes)
    validate_generation(generation)
    if type(samples_per_model) is not int or not 1 <= samples_per_model <= 8:
        raise ValueError("samples_per_model must be between 1 and 8")
    if not models or len({model["name"] for model in models}) != len(models):
        raise ValueError("models need distinct names")
    for model in models:
        validate_model(model)
    return {
        "schema": "persona-candidates-v1",
        "scenes": scenes,
        "models": models,
        "generation": generation,
        "samples_per_model": samples_per_model,
    }


def sample_candidates(scenes, models, generation, *, samples_per_model=4, call=chat_completion, on_record=None):
    """No template fallback and no auto-ranking. Failed calls are explicit records."""
    contract = sampling_contract(scenes, models, generation, samples_per_model)
    run_id, records = digest(contract), []
    for scene in scenes:
        messages = scene_messages(scene)
        for model in models:
            for index in range(samples_per_model):
                params = {**generation, "seed": generation["seed"] + index}
                request_hash = digest({"model": model, "messages": messages, "generation": params})
                record = {
                    "run_id": run_id,
                    "scene": scene,
                    "messages": messages,
                    "model": model,
                    "generation": params,
                    "sample_index": index,
                    "request_sha256": request_hash,
                    "candidate_id": digest([run_id, scene["id"], model["name"], index]),
                    "context_sha256": digest({"scene": scene, "messages": messages}),
                    "mock": False,
                }
                try:
                    record["response"] = require_text(call(model, messages, params), "generated response")
                    record["response_sha256"] = digest(record["response"])
                    record["status"] = "generated"
                except Exception as exc:
                    # Custom transports can include secrets in exception text; retain only the type.
                    record.update(status="error", error_type=type(exc).__name__)
                records.append(record)
                if on_record:
                    on_record(record)
    return records


def validate_candidate(record):
    if record.get("status") not in {"generated", "source_excerpt"} or record.get("mock") is not False:
        raise ValueError("candidate must be a successful real generation")
    if record.get("status") == "source_excerpt":
        source = record.get("source_reply", {})
        if (
            source.get("text") != record.get("response")
            or source.get("text_sha256") != digest(record.get("response"))
            or not source.get("event_ids")
            or not source.get("events_sha256")
            or source.get("scene_block_id") != record["scene"]["source_group"]
        ):
            raise ValueError("source excerpt provenance mismatch")
    messages = scene_messages(record["scene"])
    if record["messages"] != messages or record["context_sha256"] != digest(
        {"scene": record["scene"], "messages": messages}
    ):
        raise ValueError("candidate context binding mismatch")
    validate_model(record["model"])
    validate_generation(record["generation"])
    if record["request_sha256"] != digest(
        {"model": record["model"], "messages": messages, "generation": record["generation"]}
    ):
        raise ValueError("candidate request binding mismatch")
    if record["response_sha256"] != digest(require_text(record["response"], "candidate response")):
        raise ValueError("candidate response binding mismatch")
    expected = digest([record["run_id"], record["scene"]["id"], record["model"]["name"], record["sample_index"]])
    if record["candidate_id"] != expected:
        raise ValueError("candidate identity mismatch")


def validate_run(contract, records):
    """A partial journal must never silently become a complete evaluation set."""
    if contract != sampling_contract(
        contract["scenes"], contract["models"], contract["generation"], contract["samples_per_model"]
    ):
        raise ValueError("invalid sampling manifest")
    expected = {}
    run_id = digest(contract)
    for scene in contract["scenes"]:
        for model in contract["models"]:
            validate_model(model)
            for index in range(contract["samples_per_model"]):
                candidate_id = digest([run_id, scene["id"], model["name"], index])
                expected[candidate_id] = (
                    scene,
                    model,
                    {**contract["generation"], "seed": contract["generation"]["seed"] + index},
                )
    if len(records) != len(expected) or {row["candidate_id"] for row in records} != set(expected):
        raise ValueError("sampling journal is incomplete or contains duplicate/foreign candidates")
    for row in records:
        scene, model, generation = expected[row["candidate_id"]]
        if row["run_id"] != run_id or row["scene"] != scene or row["model"] != model or row["generation"] != generation:
            raise ValueError("candidate does not match sampling manifest")
