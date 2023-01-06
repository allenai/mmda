"""

UserData and HasUserData allow a Python class to have an underscore `_` property for
accessing custom fields. Data captured within this property can be serialized along with
any instance of the class.

"""
from typing import Any, Callable, Dict, Optional


class UserData:
    _data: Dict[str, Any]
    _callback: Optional[Callable[[Any], None]]

    def __init__(self, after_set_callback: Callable[[Any], None] = None):
        # Use object.__setattr__ to avoid loop on self.__setattr__ during init
        object.__setattr__(self, "_data", dict())
        object.__setattr__(self, "_callback", after_set_callback)

    def __setattr__(self, name: str, value: Any) -> None:
        if object.__getattribute__(self, name):
            raise ValueError(f"Cannot set reserved name {name}!")

        self._data.__setitem__(name, value)

        if not self._callback:
            return

        self._callback(name, value)

    def __delattr__(self, name: str) -> None:
        self._data.__delitem__(name)

    def __getattr__(self, name: str) -> Any:
        return self._data[name]

    def keys(self):
        return self._data.keys()


class HasUserData:
    _user_data: UserData

    def __init__(self, after_set_callback=None):
        self._user_data = UserData(after_set_callback=after_set_callback)

    @property
    def _(self):
        return self._user_data
