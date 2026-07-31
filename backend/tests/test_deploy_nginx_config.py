from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY_ROOT / "deploy" / "docker-compose.yml"
NGINX_SITE_CONFIG = REPOSITORY_ROOT / "deploy" / "nginx" / "nginx.conf"
VERIFY_SCRIPT = REPOSITORY_ROOT / "deploy" / "verify.sh"
BARE_START_SCRIPT = REPOSITORY_ROOT / "deploy" / "scripts" / "start_all.sh"
LEGACY_BACKEND_ENTRYPOINT = REPOSITORY_ROOT / "backend" / "main.py"


def test_nginx_http_fragment_is_mounted_under_conf_d() -> None:
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")
    site_config = NGINX_SITE_CONFIG.read_text(encoding="utf-8")

    assert "upstream fastapi_backend" in site_config
    assert "upstream vllm_backend" not in site_config
    assert "server {" in site_config
    assert "events {" not in site_config
    assert "http {" not in site_config
    assert (
        "./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro"
        in compose_text
    )
    assert "./nginx/nginx.conf:/etc/nginx/nginx.conf:ro" not in compose_text


def test_regular_api_responses_use_nginx_buffering() -> None:
    site_config = NGINX_SITE_CONFIG.read_text(encoding="utf-8")
    api_location = site_config.split("location /api/ {", 1)[1].split("}", 1)[0]

    assert "proxy_buffering off;" not in api_location


def test_auth_rate_limit_uses_the_direct_client_at_the_public_edge() -> None:
    site_config = NGINX_SITE_CONFIG.read_text(encoding="utf-8")
    standard = COMPOSE_FILE.read_text(encoding="utf-8")
    constrained = (
        REPOSITORY_ROOT / "deploy" / "docker-compose.15g.yml"
    ).read_text(encoding="utf-8")

    assert (
        "limit_req_zone $binary_remote_addr zone=auth_per_ip:10m rate=10r/m;"
        in site_config
    )
    auth_location = site_config.split(
        "location ~ ^/api/auth/(login|register)$ {", 1
    )[1].split("}", 1)[0]
    assert "client_max_body_size 8k;" in auth_location
    assert "limit_req zone=auth_per_ip burst=5 nodelay;" in auth_location
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in auth_location
    for compose_text in (standard, constrained):
        assert "AUTH_RPM" in compose_text




def test_backend_concurrency_settings_are_tunable_in_both_compose_variants() -> None:
    settings = {
        "BACKEND_LIMIT_CONCURRENCY": "256",
        "BACKEND_BACKLOG": "512",
        "BACKEND_KEEPALIVE_TIMEOUT": "10",
        "STATS_CACHE_TTL_SECONDS": "5",
        "STATS_RESOURCE_CACHE_TTL_SECONDS": "15",
        "AUTH_PASSWORD_WORKERS": "2",
        "AUTH_PASSWORD_MAX_PENDING": "8",
        "AUTH_PASSWORD_TIMEOUT_SECONDS": "10",
        "AUTH_DATABASE_WORKERS": "8",
        "AUTH_DATABASE_MAX_PENDING": "32",
        "AUTH_DATABASE_TIMEOUT_SECONDS": "5",
        "RATE_LIMIT_MAX_KEYS": "10000",
        "RATE_LIMIT_CLEANUP_INTERVAL": "64",
        "RATE_LIMIT_CLEANUP_BATCH_SIZE": "64",
        "TRAINING_EXPORT_MAX_CONCURRENCY": "2",
        "TRAINING_EXPORT_ACQUIRE_TIMEOUT": "1",
    }

    for relative_path in ("deploy/docker-compose.yml", "deploy/docker-compose.15g.yml"):
        compose_text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        for name, default in settings.items():
            assert name in compose_text
            assert f"${{{name}:-{default}}}" in compose_text



def test_all_backend_entrypoints_apply_shared_admission_limits() -> None:
    bare_start_text = BARE_START_SCRIPT.read_text(encoding="utf-8")
    legacy_entrypoint_text = LEGACY_BACKEND_ENTRYPOINT.read_text(encoding="utf-8")

    assert 'python run.py --host 0.0.0.0 --port "${BACKEND_PORT}"' in bare_start_text
    assert "python -m uvicorn app.main:app" not in bare_start_text
    for setting in (
        "BACKEND_LIMIT_CONCURRENCY",
        "BACKEND_BACKLOG",
        "BACKEND_KEEPALIVE_TIMEOUT",
    ):
        assert setting in legacy_entrypoint_text

def test_compose_lora_roots_use_their_respective_container_namespaces() -> None:
    standard = COMPOSE_FILE.read_text(encoding="utf-8")
    constrained = (
        REPOSITORY_ROOT / "deploy" / "docker-compose.15g.yml"
    ).read_text(encoding="utf-8")

    assert "./data/loras:/loras" in standard
    assert "./data/loras:/app/loras" in standard
    assert "VLLM_LORA_ROOT: /loras" in standard
    assert "LORA_PATH: /app/loras" in standard

    assert "loras:/loras:ro" in constrained
    assert "loras:/app/loras" in constrained
    assert "VLLM_LORA_ROOT=/loras" in constrained
    assert "LORA_PATH=/app/loras" in constrained


def test_vllm_healthchecks_use_the_openai_server_health_endpoint() -> None:
    standard = COMPOSE_FILE.read_text(encoding="utf-8")
    constrained = (
        REPOSITORY_ROOT / "deploy" / "docker-compose.15g.yml"
    ).read_text(encoding="utf-8")
    vllm_block = constrained.split("\n  vllm:", 1)[1].split("\n  backend:", 1)[0]
    backend_block = constrained.split("\n  backend:", 1)[1].split("\n  frontend:", 1)[0]

    assert "http://localhost:8001/health" in standard
    assert "http://localhost:8000/health" in vllm_block
    assert "http://localhost:8000/ready" not in vllm_block
    assert "http://localhost:8000/ready" in backend_block


def test_nginx_exposes_backend_readiness_without_publishing_internal_ports() -> None:
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")
    site_config = NGINX_SITE_CONFIG.read_text(encoding="utf-8")

    assert "location = /health" in site_config
    assert "location = /ready" in site_config
    assert '"80:80"' in compose_text
    assert '"8000:8000"' not in compose_text
    assert '"8001:8001"' not in compose_text
    assert '"5000:5000"' not in compose_text
    assert "proxy_set_header Host $http_host;" in site_config
    assert "proxy_set_header X-Forwarded-Host $http_host;" in site_config
    assert "proxy_set_header Host $host;" not in site_config


def test_verify_script_uses_public_entrypoint_and_avoids_persistent_test_users() -> None:
    script = VERIFY_SCRIPT.read_text(encoding="utf-8")

    assert "VERIFY_BASE_URL" in script
    assert "http://localhost:8000" not in script
    assert "http://localhost:8001" not in script
    assert "http://localhost:5000" not in script
    assert "/api/auth/register" not in script
    assert "trap cleanup EXIT" in script
    assert "-X DELETE" in script
    assert "((PASS++))" not in script


def test_nginx_preserves_outer_https_for_csrf_origin_checks() -> None:
    site_config = NGINX_SITE_CONFIG.read_text(encoding="utf-8")

    assert "map $http_x_forwarded_proto $public_scheme" in site_config
    assert "~*^https$ https;" in site_config
    assert site_config.count(
        "proxy_set_header X-Forwarded-Proto $public_scheme;"
    ) == 3
    assert "proxy_set_header X-Forwarded-Proto $scheme;" not in site_config
    assert "connect-src 'self';" in site_config
    assert "connect-src 'self' http://localhost:" not in site_config