from __future__ import annotations

import pytest
from pydantic import ValidationError

from db.schemas import LoginRequest, RegisterRequest


@pytest.mark.parametrize("request_type", [LoginRequest, RegisterRequest])
def test_auth_password_accepts_bcrypt_maximum_bytes(request_type):
    request = request_type(username="alice", password="a" * 72)

    assert request.password == "a" * 72


@pytest.mark.parametrize("request_type", [LoginRequest, RegisterRequest])
def test_auth_password_rejects_more_than_bcrypt_maximum_bytes(request_type):
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        request_type(username="alice", password="a" * 73)


@pytest.mark.parametrize("request_type", [LoginRequest, RegisterRequest])
def test_auth_password_limit_is_measured_in_utf8_bytes(request_type):
    request = request_type(username="alice", password="密" * 24)
    assert len(request.password.encode("utf-8")) == 72

    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        request_type(username="alice", password="密" * 25)


def test_login_bounds_untrusted_input_without_rejecting_legacy_short_passwords():
    request = LoginRequest(username="a", password="x")

    assert request.username == "a"
    assert request.password == "x"

    with pytest.raises(ValidationError):
        LoginRequest(username="a" * 51, password="x")

    with pytest.raises(ValidationError):
        LoginRequest(username="a", password="x" * 101)
