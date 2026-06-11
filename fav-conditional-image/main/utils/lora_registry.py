"""Registry mapping backbone names to their LoRA-injection implementations."""
from __future__ import annotations

from typing import Callable, Dict

InjectFn = Callable[..., None]
GetStateFn = Callable[..., dict]
LoadStateFn = Callable[..., None]

_INJECT_REGISTRY: Dict[str, InjectFn] = {}
_GET_STATE_REGISTRY: Dict[str, GetStateFn] = {}
_LOAD_STATE_REGISTRY: Dict[str, LoadStateFn] = {}


def register(
    backbone_name: str,
    *,
    inject: InjectFn,
    get_state: GetStateFn,
    load_state: LoadStateFn,
) -> None:
    """Register a backbone's LoRA implementation."""
    _INJECT_REGISTRY[backbone_name] = inject
    _GET_STATE_REGISTRY[backbone_name] = get_state
    _LOAD_STATE_REGISTRY[backbone_name] = load_state


def get_inject(backbone_name: str) -> InjectFn:
    if backbone_name not in _INJECT_REGISTRY:
        raise KeyError(
            f"No LoRA injector registered for backbone {backbone_name!r}. "
            f"Registered: {sorted(_INJECT_REGISTRY)}"
        )
    return _INJECT_REGISTRY[backbone_name]


def get_state_fn(backbone_name: str) -> GetStateFn:
    return _GET_STATE_REGISTRY[backbone_name]


def get_load_fn(backbone_name: str) -> LoadStateFn:
    return _LOAD_STATE_REGISTRY[backbone_name]
