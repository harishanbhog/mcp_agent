from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class SettingsConfigDict(dict):
    pass


class BaseSettings(BaseModel):
    def __init__(self, **data: Any) -> None:
        annotations = {}
        for cls in reversed(self.__class__.__mro__):
            annotations.update(getattr(cls, '__annotations__', {}))

        values = dict(data)
        for name in annotations:
            field_def = getattr(self.__class__, name, None)
            alias = field_def.alias if hasattr(field_def, 'alias') else None
            if name in values or (alias and alias in values):
                continue
            env_name = alias or name.upper()
            if env_name in os.environ:
                values[name] = os.environ[env_name]
        super().__init__(**values)
