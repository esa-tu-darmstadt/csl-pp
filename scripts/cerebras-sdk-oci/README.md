# Running the Cerebras SDK from an OCI image

The Cerebras SDK ships as a Singularity SIF image, and its own `cslc` and
`cs_python` wrappers run that SIF through Apptainer or Singularity. On Linux
that is the default and nothing here is needed: point tooling at the unpacked
SDK directory and use the wrappers it provides.

Apptainer does not exist on macOS, so this directory covers the other option:
convert the SIF's root filesystem once into a local OCI image. The image is an
ordinary OCI artifact and runs under **Apple Container**, **Docker**, or
**Podman** alike. Apple Container is only the default on Apple Silicon because
it runs the SDK's `linux/amd64` userspace under Rosetta.

| | image | runtime |
|---|---|---|
| SDK default | `.sif` | Apptainer, Singularity (Linux) |
| This directory | OCI, converted from the SIF | Apple Container, Docker, Podman |

The conversion below was validated with Apple Container 0.11 and Cerebras SDK
2.10.0. It needs roughly 20 GB of temporary space; normal use afterwards only
needs the image in the runtime's local store.

## Install and verify the SDK archive

```bash
SDK_ARCHIVE=~/Downloads/Cerebras-SDK-2.10.0-sdk-202604101435-4586d3f0d8.tar.gz
SDK_DIR=~/dev/cerebras/Cerebras-SDK-2.10.0-202604101435-4586d3f0d8

mkdir -p "$SDK_DIR"
tar -xzf "$SDK_ARCHIVE" -C "$SDK_DIR"
(cd "$SDK_DIR" && shasum -a 256 -c sha256sum.txt)

# Optional, but useful for examples and smoke tests.
tar -xzf "$SDK_DIR"/csl-extras-*.tar.gz -C "$SDK_DIR"
```

## Convert the SIF to OCI

Use a temporary amd64 Ubuntu container to locate and extract the SIF's SquashFS
partition. Extraction happens on a Linux filesystem so ownership, modes, links,
ACLs, and extended attributes are preserved.

```bash
container system start          # Apple Container only

WORK_DIR=/tmp/cerebras-sdk-2.10-oci
mkdir -p "$WORK_DIR"

container run --arch amd64 --rm --cpus 8 --memory 4G \
  --mount type=bind,source="$SDK_DIR",target=/sdk,readonly \
  --mount type=bind,source="$WORK_DIR",target=/out \
  docker.io/library/ubuntu:24.04 bash -lc '
    set -euo pipefail
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive \
      apt-get install -y singularity-container pigz >/dev/null

    sif=$(find /sdk -maxdepth 1 -name "*.sif" -print -quit)

    # `singularity sif list "$sif"` reports the SquashFS start position.
    # It is 40960 for the SDK 2.10.0 image named above.
    offset=40960

    mkdir -p /work/rootfs
    unsquashfs -o "$offset" -d /work/rootfs "$sif"
    tar --numeric-owner --xattrs --acls -C /work/rootfs -cf - . \
      | pigz -1 > /out/sdk-rootfs.tar.gz
  '
```

With Docker or Podman, replace `--arch amd64` with `--platform linux/amd64`.

Wrap the compressed root filesystem directly as an OCI layer and load it:

```bash
python3 make_oci.py \
  --layer "$WORK_DIR/sdk-rootfs.tar.gz" \
  --output "$WORK_DIR/cerebras-sdk-2.10.0.oci.tar" \
  --tag cerebras-sdk:2.10.0

container image load --input "$WORK_DIR/cerebras-sdk-2.10.0.oci.tar"
container run --arch amd64 --rm cerebras-sdk:2.10.0 uname -m
```

`docker load -i` and `podman load -i` accept the same archive. The final command
must print `x86_64`. The temporary conversion directory may then be removed;
the runtime has imported its own copy of the image.

## Run `cslc` and `cs_python`

`bin/cslc` and `bin/cs_python` in this directory are drop-in replacements for
the SDK's own wrappers. They mount the working directory at its real absolute
path so generated and referenced paths stay valid inside the container, add
mounts for absolute path arguments and for every entry in `CSL_IMPORT_PATH`,
and dispatch to the SDK's `sdk_debug_shell compile` and `python`.

```bash
export CEREBRAS_SDK_IMAGE=cerebras-sdk:2.10.0
export PATH="$PWD/bin:$PATH"
cslc layout.csl --arch=wse3 --fabric-dims=8,3 --fabric-offsets=4,1 \
  --memcpy --channels 1 -o out
```

The wrapper picks its runtime automatically: `CEREBRAS_CONTAINER_BIN` if set,
otherwise Apple Container at `/usr/local/bin/container` on macOS, otherwise the
first of `container`, `podman`, or `docker` on `PATH`. It selects the image
architecture with `--arch` for Apple Container and `--platform` for Docker and
Podman.

| variable | default |
|---|---|
| `CEREBRAS_CONTAINER_BIN` | autodetected as described above |
| `CEREBRAS_SDK_IMAGE` | **required** -- the tag chosen above |
| `CEREBRAS_SDK_CONTAINER_ARCH` | `amd64` |

`CEREBRAS_SDK_IMAGE` has no default on purpose: the tag is yours to choose
during conversion, and a built-in one would pin an SDK version and fail only
once a container is actually launched. This mirrors `CEREBRAS_SDK_SIF`, which
Apptainer has always required.

## Use from a build

csl-mlir's own integration tests (`check-csl-integration`) need `cslc` and
`cs_python`, and take the directory holding the wrappers:

```sh
-DCSL_SDK_HOST_TOOLS_DIR=<this directory>/bin
```

Any other build or script that runs `cslc` as an external program can use
`bin/cslc` directly, provided `CEREBRAS_SDK_IMAGE` is set in its environment.

Apple's documentation for amd64/Rosetta execution is in the
[Apple Container how-to](https://github.com/apple/container/blob/main/docs/how-to.md).
