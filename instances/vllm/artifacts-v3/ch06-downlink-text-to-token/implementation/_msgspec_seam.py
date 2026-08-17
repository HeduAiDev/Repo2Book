"""HOST SEAM — msgspec API subset for the ch06 subtract-only companion
(seam body identical to the ch05 companion's seam — same pin, same package gap).

The pinned vLLM (v0.27.1) serializes every cross-process message with the
`msgspec` package (`msgspec.msgpack`). That package is not installed on this
host and must not be installed from here, so this module re-implements the
exact API surface vLLM touches, backed by the `msgpack` package — the wire
bytes are therefore *genuine msgpack*, in the same wire format msgspec emits
for this usage:

- `Struct` base class with `array_like=True` (positional-array encoding —
  ALL fields ride the array; real msgspec's `omit_defaults` only trims
  map-like keys and has NO effect on positional arrays — verified against
  real msgspec 0.19.0/0.20.0/0.21.1 in the pin container), `gc=False` (no
  wire effect). The decoder additionally accepts SHORT positional arrays and
  fills missing trailing elements from field defaults (real msgspec accepts
  those too — a peer encoding fewer fields still round-trips);
- `msgspec.msgpack.Encoder(enc_hook=...)` with `.encode()` / `.encode_into()`
  (encode_into truncates the caller's bytearray to the message while keeping
  the same bytearray object — the property the engine output thread relies on);
- `msgspec.msgpack.Decoder(type=..., ext_hook=..., dec_hook=...)`;
- `msgspec.msgpack.encode` / `.decode(..., type=...)` module functions;
- `msgspec.Ext` (extension type; inline tensor payloads ride as Ext),
  `msgspec.convert`.

Known deviations from real msgspec (also in impl-notes.md):
1. inline Ext payloads are `bytes(...)`-copied once (msgpack's ExtType wants
   bytes; real msgspec passes the memoryview zero-copy) — affects only
   tensors below the 256B inline threshold;
2. the seam decoder is lenient where msgspec is strict (bool/int, unknown
   map keys), and only the type constructs the pinned vLLM structs use are
   handled;
3. mutable Struct defaults are copied per instance (real msgspec shares one
   default object; vLLM never mutates them, so behavior is identical);
4. map-like Struct encoding does NOT omit default keys (real msgspec with
   omit_defaults=True would trim them). Latent only: every wire struct in
   the pinned vLLM (EngineCoreRequest/EngineCoreOutput/EngineCoreOutputs/
   UtilityOutput) is array_like, where real msgspec also encodes all
   fields — so no map-like struct rides this chapter's wire (the dataclass
   branch is a different code path, byte-verified against the seam).

Each def/class below carries a `# SOURCE:` ref to the *vLLM call site* this
seam stands in for (msgspec itself has no vLLM source).
"""

from __future__ import annotations

import collections.abc as _cabc
import dataclasses
import enum
import types
import typing
from typing import Any

import msgpack as _msgpack

_NODEFAULT = object()  # sentinel: struct field without a default


# SOURCE: vllm/v1/serial_utils.py:L264 data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
class Ext:  # HOST SEAM of msgspec.msgpack.Ext
    """msgpack extension type (code + raw payload)."""

    # SOURCE: vllm/v1/serial_utils.py:L264 msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, ...) 构造位
    def __init__(self, code: int, data):  # data: bytes | bytearray | memoryview
        self.code = code
        self.data = data

    # SOURCE: vllm/v1/serial_utils.py:L264 (Ext 值相等 — seam 单测断言用)
    def __eq__(self, other):
        return (
            isinstance(other, Ext)
            and self.code == other.code
            and bytes(self.data) == bytes(other.data)
        )

    # SOURCE: vllm/v1/serial_utils.py:L264 (Ext 调试打印 — seam 自用)
    def __repr__(self):
        return f"Ext(code={self.code}, data=<{len(self.data)} bytes>)"


# SOURCE: vllm/v1/engine/__init__.py:L97-L102 EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False)
class _StructMeta(type):
    """Metaclass collecting positional fields + defaults from annotations."""

    # SOURCE: vllm/v1/engine/__init__.py:L97-L102 Struct 参数 (array_like/omit_defaults/gc) 消费位
    def __new__(mcs, name, bases, ns, array_like=False, omit_defaults=False, gc=True, **kw):  # noqa: N803
        cls = super().__new__(mcs, name, bases, ns, **kw)
        cls.__struct_fields__ = tuple(ns.get("__annotations__", {}).keys())
        cls.__struct_defaults__ = tuple(
            ns.get(f, _NODEFAULT) for f in cls.__struct_fields__
        )
        cls.__struct_is_array_like__ = array_like
        cls.__struct_omit_defaults__ = omit_defaults
        cls.__struct_hints__ = None  # resolved lazily at first decode
        return cls


# SOURCE: vllm/v1/engine/__init__.py:L97-L102 / L184-L188 / L230-L234 — the three wire structs
class Struct(metaclass=_StructMeta, array_like=False, omit_defaults=False, gc=True):
    """HOST SEAM of msgspec.Struct — positional fields, keyword defaults."""

    # SOURCE: vllm/v1/engine/__init__.py:L97-L102 msgspec.Struct 实例面 (positional+defaults)
    def __init__(self, *args, **kwargs):
        fields = type(self).__struct_fields__
        defaults = type(self).__struct_defaults__
        if len(args) > len(fields):
            raise TypeError(
                f"{type(self).__name__} takes at most {len(fields)} arguments"
            )
        values = list(args)
        for name, value in kwargs.items():
            if name not in fields:
                raise TypeError(f"{type(self).__name__} has no field {name!r}")
            idx = fields.index(name)
            if idx < len(values) and values[idx] is not _NODEFAULT:
                raise TypeError(
                    f"{type(self).__name__} got duplicate field {name!r}"
                )
            while len(values) <= idx:
                values.append(_NODEFAULT)
            values[idx] = value
        while len(values) < len(fields):
            values.append(_NODEFAULT)
        resolved = []
        for fname, value, default in zip(fields, values, defaults):
            if value is _NODEFAULT:
                if default is _NODEFAULT:
                    raise TypeError(
                        f"{type(self).__name__} missing field {fname!r}"
                    )
                value = _copy_default(default)
            resolved.append(value)
        for name, value in zip(fields, resolved):
            setattr(self, name, value)
        post = getattr(type(self), "__post_init__", None)
        if post is not None:
            post(self)

    # SOURCE: vllm/v1/serial_utils.py:L340 decode 还原的 Struct 比较 (msgspec 语义替身)
    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return all(
            getattr(self, f) == getattr(other, f)
            for f in type(self).__struct_fields__
        )

    # SOURCE: vllm/v1/engine/__init__.py:L97-L102 (Struct 调试打印 — seam 自用)
    def __repr__(self):
        inner = ", ".join(
            f"{f}={getattr(self, f)!r}" for f in type(self).__struct_fields__
        )
        return f"{type(self).__name__}({inner})"


# SOURCE: vllm/v1/engine/__init__.py:L97-L102 (Struct defaults machinery — HOST SEAM helper)
def _copy_default(default):  # HOST SEAM: fresh mutable defaults per instance
    if isinstance(default, list):
        return list(default)
    if isinstance(default, dict):
        return dict(default)
    if isinstance(default, set):
        return set(default)
    return default


# ── encoding ───────────────────────────────────────────────────────────────


# SOURCE: vllm/v1/serial_utils.py:L166-L178 MsgpackEncoder.encode — array_like wire form
def _encode_struct_values(obj) -> tuple:
    cls = type(obj)
    # ALL fields ride a positional array: real msgspec (verified 0.19.0/
    # 0.20.0/0.21.1 in the pin container) never omits array_like fields —
    # omit_defaults only trims map-like keys and is a no-op here. The pin's
    # wire structs pass omit_defaults=True decoratively; do NOT trim.
    # (`__struct_omit_defaults__` stays recorded for kwarg fidelity only.)
    return tuple(getattr(obj, f) for f in cls.__struct_fields__)


# SOURCE: vllm/v1/engine/core.py:L1687 ready_payload = msgspec.msgpack.encode(ready_response) — builtin mapping
def _to_builtin(obj, enc_hook=None):
    """Convert seam-known containers to plain msgpack builtins; hook the rest.

    The msgpack Packer calls this `default` recursively for every unknown
    object it meets, so hook results and nested Structs re-enter naturally.
    """
    if isinstance(obj, Struct):
        if type(obj).__struct_is_array_like__:
            return _encode_struct_values(obj)
        return {f: getattr(obj, f) for f in type(obj).__struct_fields__}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.IntEnum):
        return int(obj)
    if isinstance(obj, enum.Enum):
        # real msgspec encodes Enum members as their value (str/int enums)
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return list(obj)  # msgspec encodes sets as arrays
    if isinstance(obj, Ext):
        data = obj.data
        if not isinstance(data, bytes):
            data = bytes(data)  # msgpack ExtType wants bytes (seam deviation #1)
        return _msgpack.ExtType(obj.code, data)
    if enc_hook is not None:
        return enc_hook(obj)
    raise TypeError(
        f"Object of type {type(obj)} is not serializable. "
        "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
        "fallback to pickle-based serialization."
    )


# SOURCE: vllm/v1/serial_utils.py:L156 self.encoder = msgpack.Encoder(enc_hook=self.enc_hook) — HOST SEAM of msgspec.msgpack.Encoder
class Encoder:
    """msgpack encoder with an enc_hook for custom types."""

    # SOURCE: vllm/v1/serial_utils.py:L149-L156 msgpack.Encoder(enc_hook=...) 构造位
    def __init__(self, enc_hook=None):
        self._enc_hook = enc_hook
        # strict_types=False: msgpack-python's strict mode rejects plain
        # tuples, which both struct values and enc_hook results are (msgspec
        # always packs tuples as arrays). Unknown types still reach the
        # default hook -> TypeError, so the reject-by-default contract holds.
        self._packer = _msgpack.Packer(
            default=lambda obj: _to_builtin(obj, self._enc_hook),
            use_bin_type=True,
        )

    # SOURCE: vllm/v1/serial_utils.py:L166-L168 self.encoder.encode 调用位
    def encode(self, obj: Any) -> bytes:
        return self._packer.pack(obj)

    # SOURCE: vllm/v1/serial_utils.py:L186 self.encoder.encode_into(obj, buf) — in-place encode
    def encode_into(self, obj: Any, buf: bytearray) -> None:
        # Truncate to the message, keep the same bytearray object (the
        # property MsgpackEncoder.encode_into + the first-frame tracker rely on).
        buf[:] = self._packer.pack(obj)


# ── decoding ───────────────────────────────────────────────────────────────


# SOURCE: vllm/v1/serial_utils.py:L346 return self.decoder.decode(bufs[0]) — frame to bytes
def _as_bytes(buf) -> bytes:
    if hasattr(buf, "buffer"):  # zmq.Frame
        return bytes(buf.buffer)
    if isinstance(buf, (bytearray, memoryview)):
        return bytes(buf)
    return buf


# SOURCE: vllm/v1/serial_utils.py:L313-L336 MsgpackDecoder construction — lazy type hints
def _struct_hints(cls) -> dict:
    if cls.__struct_hints__ is None:
        cls.__struct_hints__ = typing.get_type_hints(cls)
    return cls.__struct_hints__


# SOURCE: vllm/v1/serial_utils.py:L350-L365 dec_hook(self, t, obj) contract — HOST SEAM of msgspec typed decoding
def _from_builtin(obj, t, dec_hook=None):
    if t is None or t is Any or t is object:
        return obj
    origin = typing.get_origin(t)
    if origin in (types.UnionType, typing.Union):
        if obj is None:
            return None
        for arg in typing.get_args(t):
            if arg is type(None):
                continue
            try:
                return _from_builtin(obj, arg, dec_hook)
            except (TypeError, ValueError, KeyError, AssertionError, IndexError, AttributeError):
                continue
        return obj
    if origin is list or origin is _cabc.Sequence:
        elem = typing.get_args(t)[0] if typing.get_args(t) else Any
        return [_from_builtin(x, elem, dec_hook) for x in obj]
    if origin is dict or origin is _cabc.Mapping:
        args = typing.get_args(t)
        if len(args) == 2:
            return {
                _from_builtin(k, args[0], dec_hook): _from_builtin(v, args[1], dec_hook)
                for k, v in obj.items()
            }
        return dict(obj)
    if origin is set or origin is frozenset:
        elem = typing.get_args(t)[0] if typing.get_args(t) else Any
        return {_from_builtin(x, elem, dec_hook) for x in obj}
    if origin is tuple:
        return tuple(
            _from_builtin(x, arg, dec_hook) for x, arg in zip(obj, typing.get_args(t))
        )
    if isinstance(t, type):
        if isinstance(obj, t) and not isinstance(t, (int, float, str, bool)):
            return obj
        if issubclass(t, enum.IntEnum):
            return t(obj)
        if issubclass(t, enum.Enum):
            return t(obj)
        if issubclass(t, Struct):
            values = list(obj)
            defaults = t.__struct_defaults__
            fields = t.__struct_fields__
            while len(values) < len(fields):
                values.append(_copy_default(defaults[len(values)]))
            hints = _struct_hints(t)
            values = [
                _from_builtin(v, hints.get(f, Any), dec_hook)
                for v, f in zip(values, fields)
            ]
            return t(*values)
        if dataclasses.is_dataclass(t):
            hints = typing.get_type_hints(t)
            known = {f.name for f in dataclasses.fields(t)}
            kwargs = {
                k: _from_builtin(v, hints.get(k, Any), dec_hook)
                for k, v in obj.items()
                if k in known
            }
            return t(**kwargs)
        # Custom types (torch.Tensor, np.ndarray, slice, UtilityResult, ...)
        # go through dec_hook — exactly the msgspec contract vLLM relies on.
        if dec_hook is not None:
            return dec_hook(t, obj)
        return obj
    return obj


# SOURCE: vllm/v1/serial_utils.py:L332-L334 self.decoder = msgpack.Decoder(*args, ext_hook=..., dec_hook=...) — HOST SEAM of msgspec.msgpack.Decoder
class Decoder:
    """Typed msgpack decoder with ext_hook / dec_hook."""

    # SOURCE: vllm/v1/serial_utils.py:L323-L338 msgpack.Decoder(...) 构造位
    def __init__(
        self, t: Any | None = None, *, strict: bool = True, dec_hook=None, ext_hook=None
    ):
        self._t = t
        self._dec_hook = dec_hook
        self._ext_hook = ext_hook

    # SOURCE: vllm/v1/serial_utils.py:L340-L348 decode
    def decode(self, buf):
        # SOURCE: vllm/v1/serial_utils.py:L327-L330 (构造传入 ext_hook 的 seam 面)
        def ext_hook(code, data):  # msgspec passes the payload as memoryview
            if self._ext_hook is not None:
                return self._ext_hook(code, memoryview(data))
            return Ext(code, memoryview(data))

        obj = _msgpack.unpackb(
            _as_bytes(buf), use_list=True, strict_map_key=False, ext_hook=ext_hook
        )
        return _from_builtin(obj, self._t, self._dec_hook)


# ── module functions (mirroring the msgspec.msgpack namespace) ─────────────


# SOURCE: vllm/v1/engine/core.py:L1687 msgspec.msgpack.encode(ready_response)
def encode(obj: Any) -> bytes:  # HOST SEAM of msgspec.msgpack.encode
    return Encoder().encode(obj)


# SOURCE: vllm/v1/engine/core_client.py:L743 msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
def decode(buf, *, type: Any = None) -> Any:  # noqa: A002 — mirror msgspec kwarg name
    return Decoder(t=type).decode(buf)


# SOURCE: vllm/v1/engine/core.py:L1597 msgspec.convert(v, type=p.annotation)
def convert(obj: Any, type: Any = None, dec_hook=None) -> Any:  # noqa: A002
    return _from_builtin(obj, type, dec_hook)


# Exposed under the import spellings vLLM uses (`import msgspec`,
# `from msgspec import msgpack`) — zmq_ipc binds these to those names.
seam_msgspec = types.SimpleNamespace(
    Struct=Struct,
    Ext=Ext,
    msgpack=types.SimpleNamespace(
        Encoder=Encoder,
        Decoder=Decoder,
        Ext=Ext,
        encode=encode,
        decode=decode,
    ),
    convert=convert,
)
