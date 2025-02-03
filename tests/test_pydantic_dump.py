from typing import Any, Dict, Set

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    class Config:
        extra = "allow"  # Allow arbitrary attributes

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.__dict__["_list_fields"]: Set[str] = set()
        self.__dict__["_alias"]: Dict[str, str] = {}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __getattr__(self, name):
        """Handles alias access and custom parameters."""
        if name in self._alias:
            return getattr(self, self._alias[name])

    def __setattr__(self, name, value):
        """Handles alias assignment, field setting, or adding to _param."""
        if name in self._alias:
            name = self._alias[name]
        if name in self._list_fields and not isinstance(value, list):
            value = [value]
        super().__setattr__(name, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        attr_str = "\n".join(f"    {key}: {value!r}" for key, value in attrs.items())
        return f"{self.__class__.__name__}(\n{attr_str}\n)"

    def set_alias(self, name: str, alias: str) -> None:
        self.__dict__["_alias"][alias] = name

    def ensure_list(self, name: str):
        """Mark the field to always be treated as a list"""
        value = getattr(self, name, None)
        if value is not None and not isinstance(value, list):
            setattr(self, name, [value])
        self._list_fields.add(name)


class Foo(BaseConfig):
    a: int = 1

    class Config:
        extra = "allow"


print(Foo(**{"a": 1, "b": 2}).model_dump())  # == {'a': 1, 'b': 2}

foo = Foo()
foo.b = 2
print(foo.b)
print(foo.model_dump())
