import gzip
import hashlib
import json
import os
import platform
import random
import struct
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    # Only load checkpoints created by this experiment, never untrusted files.
    return torch.load(path, map_location=device, weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class Progress:
    """Heartbeat is separate from progress; monitor displays both timestamps."""
    def __init__(self, out, jobs):
        self.path = Path(out) / "status.json"
        self.events = Path(out) / "events.jsonl"
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.last_flush = 0.0
        now = time.time()
        self.data = dict(status="running", pid=os.getpid(), started_at=now,
                         updated_at=now, progress_at=now, phase="initializing",
                         job_index=0, jobs_total=jobs, stage="starting",
                         current=0, total=0, stage_eta_seconds=None,
                         note="ETA is per stage, not total experiment time.")
        self.flush()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop_event.wait(5):
            self.flush()

    def flush(self):
        with self.lock:
            self.data["updated_at"] = time.time()
            atomic_json(self.path, self.data)
            self.last_flush = time.monotonic()

    def update(self, **kwargs):
        with self.lock:
            self.data.update(kwargs)
            self.data["progress_at"] = time.time()
            # ODE callbacks can fire thousands of times. Preserve live state without
            # turning each network evaluation into a synchronous filesystem write.
            if (time.monotonic() - self.last_flush >= 0.25 or "status" in kwargs
                    or "phase" in kwargs or kwargs.get("current") == self.data.get("total")):
                self.flush()

    def event(self, kind, **kwargs):
        record = dict(time=time.time(), event=kind, **kwargs)
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    def finish(self, status, error=None):
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.update(status=status, error=error, ended_at=time.time(),
                    stage_eta_seconds=0 if status.startswith("completed") else None)


FASHION_FILES = {
    "train-images-idx3-ubyte.gz": ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", 60000),
    "t10k-images-idx3-ubyte.gz": ("bef4ecab320f06d8554ea6380940ec79", 10000),
}


def fashion_images(root, split="train", progress=None):
    """Download official IDX images; no torchvision or labels needed."""
    root = Path(root) / "FashionMNIST" / "raw"
    root.mkdir(parents=True, exist_ok=True)
    name = ("train" if split == "train" else "t10k") + "-images-idx3-ubyte.gz"
    expected_md5, count = FASHION_FILES[name]
    path = root / name

    def valid():
        return path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected_md5

    if not valid():
        errors = []
        urls = [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/" + name,
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + name,
            "https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/" + name,
        ]
        for url in urls:
            try:
                if progress:
                    progress.update(stage="downloading Fashion-MNIST", current=0, total=1)
                    progress.event("download", url=url)
                request = urllib.request.Request(url, headers={"User-Agent": "flow-path-lab/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    with path.with_suffix(".part").open("wb") as f:
                        while True:
                            block = response.read(1 << 20)
                            if not block:
                                break
                            f.write(block)
                            if progress:
                                progress.update(current=f.tell(),
                                                total=int(response.headers.get("Content-Length", 0)))
                os.replace(path.with_suffix(".part"), path)
                if not valid():
                    raise ValueError("Fashion-MNIST checksum mismatch")
                break
            except Exception as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError(
                f"Dataset download failed. Place the official {name} in {root}. "
                + " | ".join(errors))
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if (magic, n, rows, cols) != (2051, count, 28, 28):
            raise ValueError("Invalid Fashion-MNIST IDX header")
        data = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(n, 1, 28, 28)
    return torch.from_numpy(data)


def environment_info():
    return dict(
        python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cudnn=torch.backends.cudnn.version(),
        cpu_threads=torch.get_num_threads(),
    )
