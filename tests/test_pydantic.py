from typing import Any, Dict

from pydantic import BaseModel


class BaseConfig(BaseModel):
    # Placeholder attribute to be set by child classes
    my_attribute: Dict[str, Any] = {}
    alias: Dict[str, str] = {}

    def __getattr__(self, name):
        if name in self.my_attribute:
            print("find it")
            return self.my_attribute[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        # Allow dictionary-like access to `my_attribute`
        if key in self.my_attribute:
            return self.my_attribute[key]
        if key in self.alias:
            return self.my_attribute[self.alias[key]]
        raise KeyError(f"Key '{key}' not found in my_attribute")

    def process_attribute(self):
        # Function that uses the attribute in the base class
        if not self.my_attribute:
            raise ValueError("my_attribute must be set in the child class")
        return f"Processing {self.my_attribute}"


class ChildConfig(BaseConfig):
    # Child class sets the attribute with type annotation
    my_attribute: Dict[str, int] = {"a": 1}
    alias: Dict[str, str] = {"aa": "a"}


# Example usage
child_config = ChildConfig()
print(child_config["a"])  # Output: 1
print(child_config["aa"])  # Output: 1
result = child_config.process_attribute()
print(result)  # Output: Processing {'a': 1}
