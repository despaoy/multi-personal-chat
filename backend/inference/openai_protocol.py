"""Small protocol boundaries shared by native OpenAI-compatible callers."""

from urllib.parse import urlsplit


class ProviderResponseError(RuntimeError):
    """A rejected model request, without private HTTP response/request data."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Model provider returned HTTP {status_code}")

    @property
    def http_status(self) -> int:
        # Upstream authentication must not log this application's user out.
        return 503 if self.status_code in (402, 429) else 502

    def public_detail(self) -> dict:
        if self.status_code in (401, 403):
            kind = "provider_auth_error"
            message = "模型服务认证失败，请检查模型 API 密钥及权限配置。"
        elif self.status_code == 402:
            kind = "provider_balance_error"
            message = "模型服务余额不足，请检查模型账户余额后重试。"
        elif self.status_code == 429:
            kind = "provider_rate_limited"
            message = "模型服务触发限流，请稍后重试或检查模型账户限额。"
        else:
            kind = "provider_upstream_error"
            message = f"模型服务请求失败（HTTP {self.status_code}），请检查接口、模型配置及服务状态后重试。"
        return dict(kind=kind, message=message, upstreamStatus=self.status_code)


def is_deepseek_api(base_url: str) -> bool:
    return urlsplit(base_url).hostname == "api.deepseek.com"


def validate_completion_options(
    base_url, *, top_p=None, frequency_penalty=0.0, repetition_penalty=1.0,
    enable_thinking=False, lora_name=None,
):
    """Reject requested modes that this non-thinking adapter cannot implement."""
    unsupported = []
    if lora_name is not None:
        unsupported.append("lora_name")
    if repetition_penalty != 1.0:
        unsupported.append("repetition_penalty")
    if is_deepseek_api(base_url) and frequency_penalty not in (None, 0.0):
        unsupported.append("frequency_penalty")
    if enable_thinking:
        unsupported.append("enable_thinking")
    if is_deepseek_api(base_url) and top_p not in (None, 1.0):
        unsupported.append("top_p")
    if unsupported:
        raise ValueError("Unsupported completion options: " + ", ".join(unsupported))


def nonthinking_parameters(base_url: str, *, local_template: bool = True) -> dict:
    """Use the official DeepSeek switch only for its exact service host."""
    if is_deepseek_api(base_url):
        return {"thinking": {"type": "disabled"}}
    return {"chat_template_kwargs": {"enable_thinking": False}} if local_template else {}


def completed_chat_content(payload: object) -> str:
    """Partial, filtered or reasoning-only responses are not usable answers."""
    if not isinstance(payload, dict):
        raise ValueError("invalid chat completion")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("invalid chat completion choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("chat completion did not finish normally")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat completion has no usable content")
    return content.strip()


def chat_completions_endpoint(base_url: str) -> str:
    """Accept either a service root or an already-versioned API base."""
    base = base_url.rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
