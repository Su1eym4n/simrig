"""Helpers for resolving policy checkpoints from Hugging Face Hub."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


HF_PREFIX = "hf://"


@dataclass(frozen=True)
class HuggingFacePolicyRef:
    """A policy file stored in a Hugging Face Hub repository."""

    repo_id: str
    filename: str
    revision: str | None = None


def is_huggingface_ref(value: str) -> bool:
    """Return whether a checkpoint string uses SimRig's HF URI form."""

    return value.startswith(HF_PREFIX)


def parse_huggingface_ref(value: str, *, revision: str | None = None) -> HuggingFacePolicyRef:
    """Parse ``hf://owner/repo/path/to/policy.params`` into Hub download parts."""

    if not is_huggingface_ref(value):
        raise ValueError(f"Expected Hugging Face policy ref to start with {HF_PREFIX!r}.")

    body = value[len(HF_PREFIX) :].strip("/")
    parts = [part for part in body.split("/") if part]
    if len(parts) < 3:
        raise ValueError(
            "Hugging Face policy refs must look like "
            "hf://owner/repo/path/to/policy.params."
        )
    return HuggingFacePolicyRef(
        repo_id="/".join(parts[:2]),
        filename="/".join(parts[2:]),
        revision=revision,
    )


def resolve_policy_checkpoint(
    checkpoint: Path | str,
    *,
    hf_revision: str | None = None,
    hf_token: str | None = None,
) -> Path:
    """Return a local policy checkpoint path, downloading HF refs when needed."""

    value = str(checkpoint)
    if not is_huggingface_ref(value):
        return Path(checkpoint)

    ref = parse_huggingface_ref(value, revision=hf_revision)
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face policy refs require the `huggingface_hub` package. "
            "Install SimRig with `python -m pip install -e \".[hf]\"`, or install "
            "`huggingface_hub` directly."
        ) from exc

    policy_path = PurePosixPath(ref.filename)
    config_filename = str(policy_path.with_name("config.json"))
    snapshot_path = snapshot_download(
        repo_id=ref.repo_id,
        revision=ref.revision,
        token=token,
        allow_patterns=[ref.filename, config_filename],
    )
    return Path(snapshot_path).joinpath(*policy_path.parts)
