"""Session-level model egress policy, shared by foreground and auxiliary calls."""

import ipaddress
import os
from urllib.parse import urlsplit


class LocalOnlyPolicyError(ValueError):
    def __init__(self, service="模型"):
        super().__init__(f"此会话仅允许本地{service}；请配置本地{service}服务后重试")


def is_local_only(session):
    return session.get('network_policy', session.get('baseline', {}).get('network_policy')) == 'local_only'


def local_endpoint_allowed(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            return False
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        host = parsed.hostname.lower()
        if host == 'localhost':
            return True
        try:
            if ipaddress.ip_address(host).is_loopback:
                return True
        except ValueError:
            pass
        # Explicit administrator declaration of self-hosted services; no DNS/private-IP guesses.
        for entry in os.getenv('LOCAL_MODEL_ALLOWED_ORIGINS', '').split(','):
            candidate = urlsplit(entry.strip())
            if candidate.path not in {'', '/'} or candidate.query or candidate.fragment or candidate.username or candidate.password:
                continue
            if (candidate.scheme, candidate.hostname, candidate.port or (443 if candidate.scheme == 'https' else 80)) == (parsed.scheme, host, port):
                return True
    except (TypeError, ValueError):
        return False
    return False


def require_local_endpoint(session, url, *, service="模型"):
    if is_local_only(session) and not local_endpoint_allowed(url):
        raise LocalOnlyPolicyError(service)


def require_local_provider(session, context):
    if not is_local_only(session) or context.provider_name == 'transformers_peft':
        return
    if context.provider_name == 'vllm':
        return  # Check actual load-balancer endpoints after obtaining its client.
    require_local_endpoint(session, getattr(context.provider, 'base_url', ''))
