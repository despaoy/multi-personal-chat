"""Browser reminders with strict endpoint validation and no private payload."""

import base64
import json
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import requests
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

_PROVIDERS = frozenset({"fcm.googleapis.com", "updates.push.services.mozilla.com", "web.push.apple.com"})


def decode_key(value, size):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid Web Push key")
    data = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if len(data) != size or base64.urlsafe_b64encode(data).decode().rstrip("=") != value:
        raise ValueError("Invalid Web Push key")
    return data


def validate_endpoint(endpoint):
    if not endpoint.isascii() or re.search(r"[\x00-\x20\x7f\\]", endpoint):
        raise ValueError("浏览器推送地址无效")
    url = urlsplit(endpoint)
    if (url.scheme != "https" or url.netloc not in _PROVIDERS or not url.path.strip("/") or url.fragment):
        raise ValueError("浏览器推送服务不受支持")


def validate_keys(p256dh, auth):
    ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), decode_key(p256dh, 65))
    decode_key(auth, 16)


@dataclass(frozen=True)
class WebPushConfig:
    private_key: str = field(repr=False)
    public_key: str
    subject: str

    @classmethod
    def from_env(cls):
        private = os.getenv("CHARACTER_WEB_PUSH_PRIVATE_KEY", "").strip()
        subject = os.getenv("CHARACTER_WEB_PUSH_SUBJECT", "").strip()
        if not private or not subject:
            return None
        if not subject.startswith(("mailto:", "https://")) or any(char.isspace() for char in subject):
            raise ValueError("Invalid Web Push contact URI")
        key = ec.derive_private_key(int.from_bytes(decode_key(private, 32), "big"), ec.SECP256R1())
        public = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
        return cls(private, base64.urlsafe_b64encode(public).decode().rstrip("="), subject)

    def send(self, endpoint, p256dh, auth, tag):
        from pywebpush import WebPushException, webpush

        validate_endpoint(endpoint)
        validate_keys(p256dh, auth)
        with _PushSession() as client:
            try:
                response = webpush(
                    {"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": auth}},
                    data=json.dumps({"tag": tag}), vapid_private_key=self.private_key,
                    vapid_claims={"sub": self.subject}, requests_session=client,
                    headers={"Topic": tag}, ttl=0, timeout=5,
                )
                return response.status_code
            except WebPushException as exc:
                return exc.response.status_code if exc.response is not None else 503
            except requests.RequestException:
                return 503


class _PushSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.trust_env = False

    def request(self, method, url, **kwargs):
        validate_endpoint(url)
        kwargs["allow_redirects"] = False
        return super().request(method, url, **kwargs)
