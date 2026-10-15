"""Attach a safe model failure without replacing the primary operation receipt."""

from inference.openai_protocol import ProviderResponseError


def with_provider_error(detail, error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ProviderResponseError):
            result = dict(detail) if isinstance(detail, dict) else {"message": detail}
            result["providerError"] = error.public_detail()
            return result
        error = error.__cause__
    return detail
