from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


class ConfigDict(dict):
    pass


@dataclass
class _FieldInfo:
    default: Any = None
    default_factory: Any = None
    alias: str | None = None


def Field(default: Any = None, *, default_factory: Any = None, alias: str | None = None) -> Any:
    return _FieldInfo(default=default, default_factory=default_factory, alias=alias)


class BaseModel:
    def __init__(self, **data: Any) -> None:
        annotations = {}
        for cls in reversed(self.__class__.__mro__):
            annotations.update(getattr(cls, '__annotations__', {}))

        for name in annotations:
            field_def = getattr(self.__class__, name, None)
            if isinstance(field_def, _FieldInfo):
                value = data.get(name, _missing)
                if value is _missing and field_def.alias:
                    value = data.get(field_def.alias, _missing)
                if value is _missing:
                    if field_def.default_factory is not None:
                        value = field_def.default_factory()
                    else:
                        value = copy.deepcopy(field_def.default)
            else:
                value = data.get(name, copy.deepcopy(field_def))
            setattr(self, name, value)

    @classmethod
    def model_validate(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Cannot validate value as {cls.__name__}: {value!r}")

    def model_dump(self) -> dict[str, Any]:
        annotations = {}
        for cls in reversed(self.__class__.__mro__):
            annotations.update(getattr(cls, '__annotations__', {}))

        output = {}
        for name in annotations:
            value = getattr(self, name)
            if isinstance(value, BaseModel):
                output[name] = value.model_dump()
            elif isinstance(value, list):
                output[name] = [item.model_dump() if isinstance(item, BaseModel) else item for item in value]
            else:
                output[name] = value
        return output


_missing = object()
