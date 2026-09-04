"""Scalable random access for row-aligned JSONL text banks."""
from array import array
import json
import os

import numpy as np


class TextBank:
    """Random-access JSONL backed by a persistent, incrementally-built uint64 row index."""

    @staticmethod
    def ensure_index(path, n_rows):
        """Build or extend the sidecar to ``n_rows``; callers serialize this across processes."""
        source = os.path.realpath(path)
        cache, meta_path = source + ".offsets.u64", source + ".offsets.u64.json"
        st = os.stat(source)
        identity = {"dev": st.st_dev, "ino": st.st_ino, "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns}
        indexed_rows = indexed_bytes = 0
        if os.path.exists(cache) and os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if all(meta.get(k) == v for k, v in identity.items()):
                    indexed_rows, indexed_bytes = int(meta["rows"]), int(meta["bytes"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        mode = "r+b" if os.path.exists(cache) else "w+b"
        with open(cache, mode) as offsets:
            # Discard bytes from an append that died before publishing its metadata.
            offsets.truncate(indexed_rows * np.dtype(np.uint64).itemsize)
            if indexed_rows >= n_rows:
                return
            with open(source, "rb") as texts:
                texts.seek(indexed_bytes)
                offsets.seek(0, os.SEEK_END)
                buf = array("Q")
                for line in texts:
                    buf.append(indexed_bytes)
                    indexed_bytes += len(line)
                    indexed_rows += 1
                    if len(buf) == 1_000_000:
                        buf.tofile(offsets)
                        buf = array("Q")
                    if indexed_rows == n_rows:
                        break
                if indexed_rows < n_rows:
                    raise ValueError(f"{source} has only {indexed_rows} rows; expected {n_rows}")
                if buf:
                    buf.tofile(offsets)
            offsets.flush()
            os.fsync(offsets.fileno())
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({**identity, "rows": indexed_rows, "bytes": indexed_bytes}, f)
            os.replace(tmp, meta_path)

    def __init__(self, path, n_rows, build=False):
        self.path = os.path.realpath(path)
        if build:
            self.ensure_index(self.path, n_rows)
        cache = self.path + ".offsets.u64"
        if not os.path.exists(cache) or os.path.getsize(cache) < n_rows * 8:
            raise ValueError(f"text offset cache is shorter than the requested {n_rows} rows")
        self.f = open(self.path, "rb")
        self.offsets = np.memmap(cache, dtype=np.uint64, mode="r", shape=(n_rows,))

    def get(self, row):
        self.f.seek(int(self.offsets[row]))
        return json.loads(self.f.readline())["t"]

    __getitem__ = get

    def get_many(self, start, end):
        """Read a contiguous row range with one seek."""
        self.f.seek(int(self.offsets[start]))
        return [json.loads(self.f.readline())["t"] for _ in range(end - start)]
