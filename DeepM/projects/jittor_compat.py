"""Small PyTorch-shaped compatibility surface backed entirely by Jittor.

DeepM's octree code predates this port and uses many tensor method spellings from
PyTorch.  Keeping those spellings here makes the numerical port auditable while
all tensor allocation, layers, autograd and optimization are handled by Jittor.
This module intentionally does not import PyTorch.
"""
from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import numpy as np
import jittor as _jt
from jittor import nn as _jnn
from jittor import optim as _joptim

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:  # pragma: no cover - only used without optional logging extra
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


class DataLoader:
    """Small index loader matching DeepM's custom sampler/collate contract."""
    def __init__(self, dataset, batch_size=1, sampler=None, collate_fn=None,
                 shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.collate_fn = collate_fn or (lambda batch: batch)
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = iter(self.sampler) if self.sampler is not None else iter(range(len(self.dataset)))
        for _ in range(len(self)):
            batch = [self.dataset[next(indices)] for _ in range(self.batch_size)]
            yield self.collate_fn(batch)


def _patch_var_methods():
    """Add harmless PyTorch spelling aliases missing from some Jittor releases."""
    aliases = {
        "float": lambda x: x.float32(),
        "double": lambda x: x.float64(),
        "long": lambda x: x.int64(),
        "int": lambda x: x.int32(),
        "bool": lambda x: x.bool(),
        "dim": lambda x: len(x.shape),
        "numel": lambda x: int(np.prod(x.shape)),
        "size": lambda x, dim=None: x.shape if dim is None else x.shape[dim],
        "cpu": lambda x: x,
        "detach": lambda x: x.stop_grad(),
        "contiguous": lambda x: x,
        "device": property(lambda x: "cuda" if _jt.flags.use_cuda else "cpu"),
        "is_cuda": property(lambda x: bool(_jt.flags.use_cuda)),
        "new_zeros": lambda x, shape: _jt.zeros(shape, dtype=x.dtype),
        "new_ones": lambda x, *shape: _jt.ones(shape[0] if len(shape) == 1 else shape, dtype=x.dtype),
        "new_empty": lambda x, shape: _jt.empty(shape, dtype=x.dtype),
        "new_full": lambda x, shape, value: _jt.full(shape, value, dtype=x.dtype),
        "fill_": lambda x, value: x.assign(_jt.ones(x.shape, dtype=x.dtype) * value),
        "frac": lambda x: x - _jt.floor(x),
        "masked_fill": lambda x, mask, value: _jt.where(mask, _jt.array(value).cast(x.dtype), x),
        "amin": lambda x, dim=None: x.min(dim=dim),
        "t": lambda x: x.transpose(1, 0),
        "T": property(lambda x: x.transpose(*range(len(x.shape) - 1, -1, -1))),
        "scatter_add_": lambda x, dim, index, src: _jt.scatter(x, dim, index, src, reduce="add"),
        "to": lambda x, target=None, *args, **kwargs: _cast_to(x, target),
    }
    for name, fn in aliases.items():
        if not hasattr(_jt.Var, name):
            setattr(_jt.Var, name, fn)


def _cast_to(value, target):
    if target in (_jt.float16, "float16"):
        return value.float16()
    if target in (_jt.float32, "float32"):
        return value.float32()
    if target in (_jt.float64, "float64"):
        return value.float64()
    if target in (_jt.int32, "int32"):
        return value.int32()
    if target in (_jt.int64, "int64", "long"):
        return value.int64()
    if target in (_jt.bool, "bool"):
        return value.bool()
    return value


_patch_var_methods()


class Module(_jnn.Module):
    """Jittor module accepting the original ``forward`` convention."""
    def execute(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cuda(self, device=None):
        _jt.flags.use_cuda = 1
        return self

    def to(self, device=None):
        return self

    def state_dict(self):
        return {name: value for name, value in self.named_parameters()}

    def load_state_dict(self, params, strict=True):
        self.load_parameters(params)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class ReLU(_jnn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        shape = list(x.shape)
        end = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        merged = int(np.prod(shape[self.start_dim:end + 1]))
        return x.reshape(shape[:self.start_dim] + [merged] + shape[end + 1:])


class _Init:
    @staticmethod
    def normal_(x, mean=0.0, std=1.0):
        x.assign(_jt.randn(x.shape) * std + mean)
        return x

    @staticmethod
    def constant_(x, value):
        x.assign(_jt.ones(x.shape, dtype=x.dtype) * value)
        return x

    @staticmethod
    def kaiming_normal_(x, mode="fan_out", nonlinearity="relu"):
        fan = x.shape[0] if mode == "fan_out" else int(np.prod(x.shape[1:]))
        x.assign(_jt.randn(x.shape) * np.sqrt(2.0 / max(fan, 1)))
        return x

    @staticmethod
    def zeros_(x):
        x.assign(_jt.zeros(x.shape, dtype=x.dtype))
        return x

    @staticmethod
    def ones_(x):
        x.assign(_jt.ones(x.shape, dtype=x.dtype))
        return x

    @staticmethod
    def uniform_(x, low=0.0, high=1.0):
        x.assign(_jt.rand(x.shape) * (high - low) + low)
        return x


class _Functional:
    pad = staticmethod(_jnn.pad)
    unfold = staticmethod(_jnn.unfold)
    softmax = staticmethod(_jnn.softmax)
    relu = staticmethod(_jnn.relu)

    @staticmethod
    def normalize(x, p=2, dim=1, eps=1e-12):
        denom = (abs(x) ** p).sum(dim=dim, keepdims=True) ** (1.0 / p)
        return x / _jt.maximum(denom, _jt.array(eps))


class _NN(SimpleNamespace):
    def __init__(self):
        super().__init__(
            Module=Module, ModuleList=_jnn.ModuleList, Sequential=_jnn.Sequential,
            Linear=_jnn.Linear, Conv1d=_jnn.Conv1d, Conv2d=_jnn.Conv2d,
            ConvTranspose2d=_jnn.ConvTranspose2d, BatchNorm1d=_jnn.BatchNorm1d,
            BatchNorm2d=_jnn.BatchNorm2d, GroupNorm=_jnn.GroupNorm,
            Dropout=_jnn.Dropout, ReLU=ReLU, Sigmoid=_jnn.Sigmoid,
            Flatten=Flatten, Identity=_jnn.Identity, Parameter=lambda value: value,
            CrossEntropyLoss=_jnn.CrossEntropyLoss, functional=_Functional,
            init=_Init, utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: None),
        )


class _Cuda:
    @staticmethod
    def is_available():
        try:
            return bool(_jt.compiler.has_cuda)
        except Exception:
            return False

    @staticmethod
    def set_device(device):
        _jt.flags.use_cuda = 1

    current_device = staticmethod(lambda: 0)
    manual_seed = staticmethod(lambda seed: _jt.set_global_seed(seed))
    manual_seed_all = manual_seed
    empty_cache = staticmethod(lambda: _jt.clean())
    memory_allocated = staticmethod(lambda *args: 0)
    memory_reserved = staticmethod(lambda *args: 0)


class _Checkpoint:
    @staticmethod
    def checkpoint(function, *args, **kwargs):
        # Jittor rematerializes/fuses its lazy graph; an eager wrapper preserves
        # values and gradients without depending on torch checkpoint semantics.
        return function(*args)


class _Sparse:
    @staticmethod
    def mm(matrix, dense):
        return matrix @ dense


class _Backends:
    cudnn = SimpleNamespace(benchmark=False, deterministic=True)


class _TensorMeta(type):
    def __call__(cls, *args):
        if len(args) > 1 and all(isinstance(value, int) for value in args):
            return _jt.empty(args)
        if len(args) == 1:
            return _jt.array(args[0])
        return _jt.empty(args)

    def __instancecheck__(cls, instance):
        return isinstance(instance, _jt.Var)


class Tensor(metaclass=_TensorMeta):
    pass


class _TorchFacade:
    Tensor = Tensor
    nn = _NN()
    optim = SimpleNamespace(SGD=_joptim.SGD, Adam=_joptim.Adam, AdamW=_joptim.AdamW)
    cuda = _Cuda()
    sparse = _Sparse()
    backends = _Backends()
    utils = SimpleNamespace(
        checkpoint=_Checkpoint(),
        data=SimpleNamespace(DataLoader=DataLoader),
        tensorboard=SimpleNamespace(SummaryWriter=SummaryWriter),
    )
    float16 = _jt.float16
    float32 = _jt.float32
    float64 = _jt.float64
    int32 = _jt.int32
    int64 = _jt.int64
    long = _jt.int64
    bool = _jt.bool
    __version__ = getattr(_jt, "__version__", "1.3")
    no_grad = staticmethod(_jt.no_grad)

    @staticmethod
    def _shape(value):
        if isinstance(value, _jt.Var):
            return int(value.item())
        if isinstance(value, (tuple, list)):
            return type(value)(_TorchFacade._shape(item) for item in value)
        return value

    @staticmethod
    def _factory(function, *args, **kwargs):
        kwargs.pop("device", None)
        args = tuple(_TorchFacade._shape(arg) for arg in args)
        return function(*args, **kwargs)

    zeros = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.zeros, *a, **k))
    ones = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.ones, *a, **k))
    empty = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.empty, *a, **k))
    full = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.full, *a, **k))
    arange = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.arange, *a, **k))
    rand = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.rand, *a, **k))
    randn = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.randn, *a, **k))
    randint = staticmethod(lambda *a, **k: _TorchFacade._factory(_jt.randint, *a, **k))
    mm = staticmethod(lambda left, right: left @ right)

    @staticmethod
    def meshgrid(*args, **kwargs):
        kwargs.pop("indexing", None)
        return _jt.meshgrid(*args, **kwargs)

    @staticmethod
    def unique(value, sorted=True, return_inverse=False, return_counts=False, dim=None):
        return _jt.unique(value, return_inverse=return_inverse,
                          return_counts=return_counts, dim=dim)

    @staticmethod
    def unique_consecutive(value, return_inverse=False, return_counts=False, dim=None):
        # O-CNN invokes this after sorting parent keys, so jt.unique has the
        # same ordering and inverse/count behavior here.
        return _jt.unique(value, return_inverse=return_inverse,
                          return_counts=return_counts, dim=dim)

    @staticmethod
    def bucketize(value, boundaries, right=False):
        return _jt.searchsorted(boundaries, value, right=right)

    @staticmethod
    def tensor(data, dtype=None, device=None, requires_grad=False):
        out = _jt.array(data, dtype=dtype)
        if requires_grad:
            out.start_grad()
        return out

    from_numpy = staticmethod(lambda x: _jt.array(np.asarray(x)))
    manual_seed = staticmethod(lambda seed: _jt.set_global_seed(seed))
    device = staticmethod(lambda name=None: name or ("cuda" if _Cuda.is_available() else "cpu"))
    save = staticmethod(_jt.save)

    @staticmethod
    def load(path, map_location=None):
        return _jt.load(path)

    @staticmethod
    def sparse_coo_tensor(indices, values, size, device=None):
        out = _jt.zeros(size, dtype=values.dtype)
        coords = tuple(indices[i] for i in range(indices.shape[0]))
        out[coords] = values
        return out

    @staticmethod
    def use_deterministic_algorithms(enabled):
        return None

    def __getattr__(self, name):
        if hasattr(_jt, name):
            return getattr(_jt, name)
        raise AttributeError(f"Jittor compatibility layer has no operation {name!r}")


torch = _TorchFacade()
jt = torch
nn = torch.nn
F = _Functional
Function = _jt.Function
