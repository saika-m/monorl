"""The keys of a free-form ``params`` mapping that a built-in component reads.

``rewards[].params`` and ``env.params`` are mappings the schema cannot type: what they may hold
depends on the ``type`` next to them. A built-in type therefore declares its keys next to the code
that reads them (:func:`declare_params`) and reads them through :func:`checked_params`, which
refuses a mapping holding any other key and, the other way round, a read of a key that was not
declared. The schema asks :func:`unknown_params` when a config is validated, so a misspelt key is
reported before anything is built. A type that declares nothing (one registered through
``providers:``) is not restricted.

Imports nothing heavy: the declarations of a namespace are loaded on demand, by importing the
built-in provider of the namespace and, for a name an adapter owns, the adapter's provider.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Iterator, Mapping
from typing import Any

from monorl.core.errors import ConfigError

_PROVIDERS = {"reward": "monorl.rewards", "env": "monorl.envs"}
_KINDS = {"reward": "reward", "env": "environment"}
_declared: dict[tuple[str, str], tuple[str, ...]] = {}

Loc = tuple[str | int, ...]


def declare_params(namespace: str, name: str, *keys: str) -> tuple[str, ...]:
    """Declare ``keys`` as everything the built-in ``name`` of ``namespace`` (``reward`` or ``env``)
    reads from its ``params``: in its factory and constructor, later at runtime, and in every
    backend that reads the same mapping."""
    if namespace not in _PROVIDERS:
        raise ValueError(f"params are declared for {sorted(_PROVIDERS)}, not for {namespace!r}")
    _declared[(namespace, name)] = keys
    return keys


def declared_params(namespace: str, name: str) -> tuple[str, ...] | None:
    """The declared keys of ``name``, or ``None`` for a type that declares none (a user's)."""
    if (namespace, name) not in _declared:
        modules = [_PROVIDERS[namespace]]
        if namespace == "env":
            from monorl.adapters import provider_for_name

            modules.append(provider_for_name(name))
        for module in modules:
            if module is not None and (namespace, name) not in _declared:
                try:
                    importlib.import_module(module)
                except ImportError:  # a provider that is absent declares nothing
                    continue
    return _declared.get((namespace, name))


def _problem(namespace: str, name: str, key: str, known: tuple[str, ...]) -> str:
    close = difflib.get_close_matches(key, known, n=1)
    takes = f"he {name} {_KINDS[namespace]} takes: {', '.join(known) or '(no params)'}"
    return f"unknown key; did you mean {close[0]}? T{takes}" if close else f"unknown key; t{takes}"


def unknown_params(namespace: str, name: str, params: Any) -> list[tuple[Loc, str]]:
    """What is wrong with the keys of ``params`` for the built-in ``name``: ``(loc, problem)`` per
    unknown key, ``loc`` being the key's path below ``params``. The members of a ``composite``
    reward are looked at too. Empty for a type that declares nothing."""
    known = declared_params(namespace, name)
    if known is None or not isinstance(params, Mapping):
        return []
    problems: list[tuple[Loc, str]] = [
        ((str(key),), _problem(namespace, name, str(key), known)) for key in params if key not in known
    ]
    if (namespace, name) == ("reward", "composite") and isinstance(params.get("rewards"), (list, tuple)):
        for i, member in enumerate(params["rewards"]):
            if not isinstance(member, Mapping):  # a RewardSpec
                member = {"type": getattr(member, "type", None), "params": getattr(member, "params", None)}
            problems += [
                (("rewards", i, "params", *loc), problem)
                for loc, problem in unknown_params("reward", str(member.get("type")), member.get("params"))
            ]
    return problems


class Params(Mapping[str, Any]):
    """The ``params`` of a built-in, as :func:`checked_params` returns them: a read-only view that
    holds declared keys only and refuses to be asked for any other."""

    def __init__(self, namespace: str, name: str, data: Mapping[str, Any], known: tuple[str, ...]) -> None:
        self._owner = f"{name} {_KINDS[namespace]}"
        self._data = data
        self._known = known

    def __getitem__(self, key: str) -> Any:
        if key not in self._known:
            # not a KeyError: `get` and `in` would take that for an absent key
            raise RuntimeError(
                f"the {self._owner} reads params.{key}, which it does not declare; add the key to "
                "its declare_params(...)"
            )
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def checked_params(namespace: str, name: str, params: Mapping[str, Any], *, where: str = "params") -> Params:
    """``params`` of the built-in ``name`` for reading. Raises :class:`ConfigError` when they hold
    a key that ``name`` does not declare: one line per key, ``<where>.<key>: unknown key`` with
    the key it resembles and the keys there are."""
    if isinstance(params, Params):
        return params
    known = _declared[(namespace, name)]
    unknown = [key for key in params if key not in known]
    if unknown:
        raise ConfigError(
            "\n".join(f"{where}.{key}: {_problem(namespace, name, str(key), known)}" for key in unknown)
        )
    return Params(namespace, name, params, known)
