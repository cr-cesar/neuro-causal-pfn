"""Keyed hashing of image file names.

The names of the lesion images embed clinical fields, so nothing that leaves
the cluster may carry them. A file name is replaced by the first 16 hex
digits of sha256(salt + "|" + base name without the NIfTI extension). The
salt lives in a 0600 file on the cluster (``HASH_SALT_FILE``, default
``~/.cache_salt``) and is never bundled; two exports made with the same salt
(caches, latents, images, fold tables) therefore agree on every name.
"""
from __future__ import annotations

import hashlib
import os
import secrets

NIFTI_EXTS = (".nii.gz", ".nii")


def load_salt(path: str | None = None) -> str:
    """The salt of ``path`` (default ``$HASH_SALT_FILE`` or ``~/.cache_salt``);
    a missing file is created with 32 random bytes and mode 0600."""
    path = os.path.expanduser(path or os.environ.get("HASH_SALT_FILE", "~/.cache_salt"))
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(path, 0o600)
    with open(path) as f:
        salt = f.read().strip()
    if len(salt) < 16:
        raise SystemExit("the salt file is too short")
    return salt


def split_nifti_ext(name: str) -> tuple[str, str]:
    base = os.path.basename(str(name).replace("\\", "/"))
    for ext in NIFTI_EXTS:
        if base.lower().endswith(ext):
            return base[: -len(ext)], base[-len(ext):]
    return base, ""


def hash_name(name: str, salt: str) -> str:
    """16 hex digits for a file name (directory and NIfTI extension ignored)."""
    base, _ = split_nifti_ext(name)
    return hashlib.sha256((salt + "|" + base).encode()).hexdigest()[:16]


def hash_filename(name: str, salt: str) -> str:
    """The hashed name with the source's NIfTI extension kept."""
    return hash_name(name, salt) + split_nifti_ext(name)[1]


def hash_token(token: str, salt: str) -> str:
    """16 hex digits for any other identifier (a group id)."""
    return hashlib.sha256((salt + "|grp|" + str(token)).encode()).hexdigest()[:16]
