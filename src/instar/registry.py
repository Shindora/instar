from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Registry[T]:
    """Named plug-ins of one kind, registered with a decorator."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def __call__(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items and not _same_definition(self._items[name], obj):
                raise ValueError(f"duplicate {self.kind}: {name!r}")
            self._items[name] = obj
            return obj

        return decorator

    def __getitem__(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown {self.kind}: {name!r} (known: {known})") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)


def _same_definition(old: object, new: object) -> bool:
    """A module imported again re-registers the same names; that is a reload, not a clash.
    An entry that wraps a function (`.fn`) is compared by that function."""
    old, new = getattr(old, "fn", old), getattr(new, "fn", new)
    where = ("__module__", "__qualname__")
    return all(getattr(old, a, None) == getattr(new, a, object()) for a in where)
