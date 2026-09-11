from typing import Any
import sys

from pydantic import BaseModel, Field, model_validator

TYPES = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "array": "list",
    "object": "dict"
}


class FuncDef(BaseModel):

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=300)
    params: dict[str, str] = Field(...)
    returns: Any = Field(default=None)

    @model_validator(mode='before')
    @classmethod
    def set_params(cls, data: dict[str, Any | dict]) -> dict[str, str | dict]:  # noqa: E501
        data["params"] = {}
        for name, pack in data["parameters"].items():
            data["params"][name] = TYPES[pack["type"]]
        return data

    @model_validator(mode='after')
    def check_coherence(self) -> Any:
        if len(self.params) < 1:
            sys.exit(f"Not enough params for \'{self.name}\'")
        return self
