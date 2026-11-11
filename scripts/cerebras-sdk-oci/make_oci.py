#!/usr/bin/env python3
"""Wrap a gzip-compressed rootfs tarball in an OCI image archive.

The input archive is used directly as the image's single compressed layer.  In
particular, this avoids routing multi-gigabyte SDK files through a container
builder and avoids a second compression pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


CHUNK_SIZE = 8 * 1024 * 1024
OCI_IMAGE_CONFIG = "application/vnd.oci.image.config.v1+json"
OCI_IMAGE_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
OCI_IMAGE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_uncompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, fileobj=io.BytesIO(data))


def add_layer(archive: tarfile.TarFile, path: Path, digest: str) -> None:
    info = archive.gettarinfo(str(path), arcname=f"blobs/sha256/{digest}")
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--architecture", default="amd64")
    args = parser.parse_args()

    layer = args.layer.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")

    layer_digest = sha256_file(layer)
    diff_id = sha256_uncompressed_gzip(layer)

    config = json_bytes(
        {
            "architecture": args.architecture,
            "config": {
                "Entrypoint": ["/.singularity.d/actions/exec"],
                "WorkingDir": "/",
            },
            "history": [
                {
                    "created_by": "csl-mlir Cerebras SDK SIF-to-OCI conversion",
                }
            ],
            "os": "linux",
            "rootfs": {
                "diff_ids": [f"sha256:{diff_id}"],
                "type": "layers",
            },
        }
    )
    config_digest = hashlib.sha256(config).hexdigest()

    manifest = json_bytes(
        {
            "config": {
                "digest": f"sha256:{config_digest}",
                "mediaType": OCI_IMAGE_CONFIG,
                "size": len(config),
            },
            "layers": [
                {
                    "digest": f"sha256:{layer_digest}",
                    "mediaType": OCI_IMAGE_LAYER,
                    "size": layer.stat().st_size,
                }
            ],
            "mediaType": OCI_IMAGE_MANIFEST,
            "schemaVersion": 2,
        }
    )
    manifest_digest = hashlib.sha256(manifest).hexdigest()

    index = json_bytes(
        {
            "manifests": [
                {
                    "annotations": {
                        "org.opencontainers.image.ref.name": args.tag,
                    },
                    "digest": f"sha256:{manifest_digest}",
                    "mediaType": OCI_IMAGE_MANIFEST,
                    "platform": {
                        "architecture": args.architecture,
                        "os": "linux",
                    },
                    "size": len(manifest),
                }
            ],
            "schemaVersion": 2,
        }
    )

    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with tarfile.open(temporary, "w") as archive:
            add_bytes(archive, "oci-layout", b'{"imageLayoutVersion":"1.0.0"}')
            add_bytes(archive, "index.json", index)
            add_bytes(archive, f"blobs/sha256/{config_digest}", config)
            add_bytes(archive, f"blobs/sha256/{manifest_digest}", manifest)
            add_layer(archive, layer, layer_digest)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"wrote {output}")
    print(f"layer digest: sha256:{layer_digest}")
    print(f"layer diff_id: sha256:{diff_id}")


if __name__ == "__main__":
    main()
