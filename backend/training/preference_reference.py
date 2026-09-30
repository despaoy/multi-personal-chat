"""Explicit SFT reference adapter integrity checks, without copying the base model."""

from __future__ import annotations

import hashlib


def adapter_parameters(model, name):
    marker = f".{name}."
    return {key.replace(marker, ".<adapter>."): value for key, value in model.named_parameters() if marker in key}


def assert_initial_reference(model):
    import torch

    policy = adapter_parameters(model, "default")
    reference = adapter_parameters(model, "reference")
    if not policy or policy.keys() != reference.keys():
        raise RuntimeError("SFT policy/reference adapter parameter sets differ")
    for name, value in policy.items():
        if not torch.equal(value.detach(), reference[name].detach()):
            raise RuntimeError("SFT policy/reference adapters do not start at identical weights")
    return reference_fingerprint(model)


def reference_fingerprint(model):
    parameters = adapter_parameters(model, "reference")
    if not parameters or any(value.requires_grad for value in parameters.values()):
        raise RuntimeError("reference adapter must exist and remain frozen")
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str((value.shape, value.dtype)).encode("utf-8"))
        # Byte view also supports bfloat16, which NumPy cannot represent directly.
        import torch

        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_reference_unchanged(model, expected):
    if reference_fingerprint(model) != expected:
        raise RuntimeError("frozen SFT reference adapter changed during preference training")
