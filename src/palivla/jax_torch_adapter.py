# jax_torch_adapter.py
from __future__ import annotations
from typing import Any, Iterable, Iterator, Sequence, Optional, Callable
import threading, queue, numpy as np, jax, jax.numpy as jnp

# ---- background prefetch helper ----
class _BackgroundPrefetch(Iterator):
    def __init__(self, it: Iterable, prefetch: int = 2):
        self._it = iter(it)
        self._q: "queue.Queue[object]" = queue.Queue(prefetch)
        self._sentinel = object()
        def _loop():
            try:
                for x in self._it:
                    self._q.put(x)
            finally:
                self._q.put(self._sentinel)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
    def __iter__(self): return self
    def __next__(self):
        x = self._q.get()
        if x is self._sentinel: raise StopIteration
        return x

# ---- pytree utilities ----
def _to_numpy(x: Any) -> Any:
    import torch
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)): return type(x)(_to_numpy(v) for v in x)
    if isinstance(x, dict): return {k: _to_numpy(v) for k, v in x.items()}
    return x

def _stack(samples: Sequence[Any]) -> Any:
    s0 = samples[0]
    if isinstance(s0, dict): return {k: _stack([s[k] for s in samples]) for k in s0}
    if isinstance(s0, (list, tuple)):
        cols = list(zip(*samples))
        return type(s0)(_stack(list(c)) for c in cols)
    return np.stack(samples, axis=0)

def _device_put_tree(x: Any) -> Any:
    if isinstance(x, dict): return {k: _device_put_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_device_put_tree(v) for v in x)
    return jax.device_put(jnp.asarray(x))

def _shard_and_put(x: Any, devices: Sequence[jax.Device]) -> Any:
    n = len(devices)
    def _split_put(arr: np.ndarray):
        if arr.shape[0] % n != 0:
            arr = arr[: arr.shape[0] - (arr.shape[0] % n)]
        parts = np.split(arr, n, axis=0)
        return jax.device_put_sharded([jnp.asarray(p) for p in parts], devices)
    if isinstance(x, dict): return {k: _shard_and_put(v, devices) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_shard_and_put(v, devices) for v in x)
    return _split_put(x)
def worker_init_fn(worker_id):
    import fsspec
    global fs
    fs = fsspec.filesystem("gcs", asynchronous=False)
# ---- the adapter ----
class TorchToJaxDataset:
    """
    Wrap a torch.utils.data.Dataset so it looks like your existing dataset:
        TorchToJaxDataset(dataset, sampler=..., num_workers=8).batch(B).iterator()
    """
    def __init__(
        self,
        dataset: "torch.utils.data.Dataset",
        sampler: Optional["torch.utils.data.Sampler"] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        collate_override: Optional[Callable[[list], Any]] = None,
        seed: Optional[int] = None,
    ):
        import torch
        self.dataset = dataset
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (persistent_workers
                                   if persistent_workers is not None
                                   else num_workers > 0)
        self._generator = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            self._generator = g
        self._collate = collate_override

    class _Batched:
        def __init__(
            self, outer: "TorchToJaxDataset",
            batch_size: int,
            drop_last: bool = True,
            shuffle: bool = False,
            shard_for_pmap: bool = False,
            prefetch_host: int = 2,
            prefetch_device: int = 2,
        ):
            self.outer = outer
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.shard_for_pmap = shard_for_pmap
            self.prefetch_host = prefetch_host
            self.prefetch_device = prefetch_device

        def iterator(self) -> Iterator[Any]:
            import torch
            from torch.utils.data import DataLoader
            collate = self.outer._collate or (lambda samples: _stack([_to_numpy(s) for s in samples]))

            loader = DataLoader(
                self.outer.dataset,
                batch_size=self.batch_size,
                sampler=self.outer.sampler,
                shuffle=(self.outer.sampler is None and self.shuffle),
                num_workers=self.outer.num_workers,
                pin_memory=self.outer.pin_memory,
                drop_last=self.drop_last,
                persistent_workers=self.outer.persistent_workers,
                collate_fn=collate,
                generator=self.outer._generator,
                worker_init_fn=worker_init_fn
            )

            host_prefetch = _BackgroundPrefetch(loader, prefetch=self.prefetch_host)
            devices = jax.local_devices()

            def _to_device():
                for batch_np in host_prefetch:
                    yield (_shard_and_put(batch_np, devices)
                           if self.shard_for_pmap else
                           _device_put_tree(batch_np))
            return _BackgroundPrefetch(_to_device(), prefetch=self.prefetch_device)

    def batch(
        self,
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle: bool = False,
        shard_for_pmap: bool = False,
        prefetch_host: int = 2,
        prefetch_device: int = 2,
    ) -> "_Batched":
        return TorchToJaxDataset._Batched(
            self,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            shard_for_pmap=shard_for_pmap,
            prefetch_host=prefetch_host,
            prefetch_device=prefetch_device,
        )