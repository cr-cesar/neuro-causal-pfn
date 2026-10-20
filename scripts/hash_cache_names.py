"""Copy prior caches, fold pools and latent files with their file names hashed.

The training of the Neuro-Causal-PFN reads anatomy rows and latents from npz
caches; it never needs the image file names, but the caches carry them in a
``files`` array, and the names of the lesion images embed clinical fields.
Before a cache leaves the cluster, this script writes a copy in which every
name is replaced by a keyed hash (sha256 of salt + base name, 16 hex digits),
so that rows can still be matched between files made with the same salt
while the names themselves stay on the cluster::

    HASH_SALT_FILE=~/.cache_salt python scripts/hash_cache_names.py \\
        --src outputs/prior_cache outputs/latents_pf_bcb --dst ~/Scratch/bundle/outputs

Every ``*.npz`` under each source is copied to the same relative path under
``--dst``; keys named ``files`` (any array of strings) are hashed, every
other key is copied unchanged, and any string key naming a directory
(``images_dir``, ``fold_file``) is reduced to its base name. Prints counts
only. The salt is read from HASH_SALT_FILE (created with random bytes when
missing) and must never be bundled.
"""
import argparse
import hashlib
import os
import secrets
import sys

import numpy as np

STRING_PATH_KEYS = ("images_dir", "fold_file")


def load_salt(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(path, 0o600)
    with open(path) as f:
        salt = f.read().strip()
    if len(salt) < 16:
        sys.exit("the salt file is too short")
    return salt


def hash_name(name: str, salt: str) -> str:
    base = os.path.basename(str(name).replace("\\", "/"))
    for ext in (".nii.gz", ".nii"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return hashlib.sha256((salt + "|" + base).encode()).hexdigest()[:16]


def hash_npz(src: str, dst: str, salt: str) -> int:
    with np.load(src, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    n = 0
    for k, v in list(data.items()):
        if k == "files":
            data[k] = np.array([hash_name(x, salt) for x in np.asarray(v).ravel()]).reshape(np.shape(v))
            n = int(np.size(v))
        elif k in STRING_PATH_KEYS and np.ndim(v) == 0:
            parts = [os.path.basename(p.strip()) for p in str(np.asarray(v).item()).split("+")]
            data[k] = np.array(" + ".join(p for p in parts if p))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez_compressed(dst, **data)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="files or directories (searched for *.npz)")
    ap.add_argument("--dst", required=True, help="destination root")
    ap.add_argument("--salt-file", default=os.environ.get("HASH_SALT_FILE", "~/.cache_salt"))
    args = ap.parse_args(argv)
    salt = load_salt(args.salt_file)
    files, n_names = 0, 0
    for src in args.src:
        src = src.rstrip("/")
        if os.path.isdir(src):
            for root, _, names in os.walk(src):
                for nm in sorted(names):
                    if nm.endswith(".npz"):
                        rel = os.path.relpath(os.path.join(root, nm), os.path.dirname(src))
                        n_names += hash_npz(os.path.join(root, nm), os.path.join(args.dst, rel), salt)
                        files += 1
        elif src.endswith(".npz"):
            n_names += hash_npz(src, os.path.join(args.dst, os.path.basename(src)), salt)
            files += 1
        else:
            sys.exit(f"{src}: not an npz file or a directory")
    print(f"{files} npz files copied to {args.dst}, {n_names} file names hashed")


if __name__ == "__main__":
    main()
