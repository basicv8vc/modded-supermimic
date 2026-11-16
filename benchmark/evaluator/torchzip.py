"""Read ``torch.save`` zip checkpoints without importing torch.

A torch zip checkpoint is a pickle (``data.pkl``) describing an OrderedDict of
tensors plus one raw little-endian buffer per storage under ``data/<key>``.
That is enough to rebuild every tensor with numpy, which matters because the
canonical evaluator is expected to run in an environment without torch.

``scripts/read_torch_ckpt.py`` is the CLI wrapper around this module.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": "bf16",
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class TorchZipError(RuntimeError):
    """The file is not a torch zip checkpoint we can decode."""


class _StorageRef:
    __slots__ = ("dtype_name", "key", "numel")

    def __init__(self, dtype_name: str, key: str, numel: int):
        self.dtype_name = dtype_name
        self.key = key
        self.numel = numel


class _TensorRef:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *args, **kwargs):
    return _TensorRef(storage, storage_offset, size, stride)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in DTYPES:
            return name
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        if module.startswith("torch"):
            return lambda *a, **k: None
        return super().find_class(module, name)

    def persistent_load(self, pid):
        if pid[0] != "storage":
            raise TorchZipError(f"unsupported persistent id {pid[0]!r}")
        return _StorageRef(pid[1], pid[2], pid[4])


def _decode_bf16(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    """Return ``{tensor_name: ndarray}`` for every tensor in the checkpoint."""
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise TorchZipError(f"{path} is not a torch zip checkpoint") from exc

    with zf:
        names = zf.namelist()
        if not names:
            raise TorchZipError(f"{path} is empty")
        prefix = names[0].split("/")[0]
        try:
            with zf.open(f"{prefix}/data.pkl") as fh:
                obj = _Unpickler(fh).load()
        except KeyError as exc:
            raise TorchZipError(f"{path} has no data.pkl") from exc

        blobs: dict[str, bytes] = {}

        def blob(key: str) -> bytes:
            if key not in blobs:
                blobs[key] = zf.read(f"{prefix}/data/{key}")
            return blobs[key]

        out: dict[str, np.ndarray] = {}

        def materialize(ref: _TensorRef) -> np.ndarray:
            st = ref.storage
            dtype = DTYPES.get(st.dtype_name)
            if dtype is None:
                raise TorchZipError(f"unsupported storage type {st.dtype_name!r}")
            raw = blob(st.key)
            flat = _decode_bf16(raw) if dtype == "bf16" else np.frombuffer(raw, dtype=dtype)
            if not ref.size:
                return flat[ref.offset : ref.offset + 1].copy()
            count = int(np.prod(ref.size))
            if count == len(flat) - ref.offset:
                return flat[ref.offset : ref.offset + count].reshape(ref.size).copy()
            strides = [s * flat.itemsize for s in ref.stride]
            return np.lib.stride_tricks.as_strided(
                flat[ref.offset :], shape=ref.size, strides=strides
            ).copy()

        def walk(node, name=""):
            if isinstance(node, _TensorRef):
                out[name] = materialize(node)
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{name}.{k}" if name else str(k))
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    walk(v, f"{name}.{i}" if name else str(i))

        walk(obj)
        return out
