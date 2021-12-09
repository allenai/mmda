from typing import Any, Callable, Dict, Iterable


class UserData:
    _data: Dict[str, Any]
    _callback: Callable

    def __init__(self, after_set_callback=None):
        # Use object.__setattr__ to avoid loop on self.__setattr__ during init
        object.__setattr__(self, "_data", dict())
        object.__setattr__(self, "_callback", after_set_callback)

    def __setattr__(self, name: str, value: Any) -> None:
        self._data.__setitem__(name, value)

        if self._callback is None:
            return

        if isinstance(value, Iterable):
            for v in value:
                self._callback(name, v)
        else:
            self._callback(name, value)

    def __delattr__(self, name: str) -> None:
        self._data.__delitem__(name)

    def __getattr__(self, name: str) -> Any:
        return self._data[name]


class UserDataMixin:
    _user_data: UserData

    def __init__(self, after_set_callback=None):
        self._data = UserData(after_set_callback=after_set_callback)

    @property
    def _(self):
        return self._data
