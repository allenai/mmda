from copy import deepcopy
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

__all__ = ["store_field_in_metadata", "Metadata"]


class __DEFAULT:
    # object to keep track if a default value is provided when getting
    # or popping a key from Metadata
    ...


class Metadata:
    """An object that contains metadata for an annotation.
    It supports dot access and dict-like access."""

    @overload
    def get(self, key: str) -> Any:
        """Get value with name `key` in metadata;
        raise `KeyError` if not found"""
        ...

    @overload
    def get(self, key: str, default: Any) -> Any:
        """Get value with name `key` in metadata;
        return `default` if not found"""
        ...

    def get(self, key: str, default: Optional[Any] = __DEFAULT) -> Any:
        """Get value with name `key` in metadata;
        if not found, return `default` if specified,
        otherwise raise `KeyError`"""
        if key in self.__dict__:
            return self.__dict__[key]
        elif default != __DEFAULT:
            return default
        else:
            raise KeyError(f"{key} not found in metadata")

    def has(self, key: str) -> bool:
        """Check if metadata contains key `key`; return `True` if so,
        `False` otherwise"""
        return key in self.__dict__

    def set(self, key: str, value: Any) -> None:
        """Set `key` in metadata to `value`; key must be a valid Python
        identifier (that is, a valid variable name) otherwise,
        raise a ValueError"""
        if not key.isidentifier():
            raise ValueError(
                f"`{key}` is not a valid variable name, "
                "so it cannot be used as key in metadata"
            )
        self.__dict__[key] = value

    @overload
    def pop(self, key: str) -> Any:
        """Remove & returns value for `key` from metadata;
        raise `KeyError` if not found"""
        ...

    @overload
    def pop(self, key: str, default: Any) -> Any:
        """Remove & returns value for `key` from metadata;
        if not found, return `default`"""
        ...

    def pop(self, key: str, default: Optional[Any] = __DEFAULT) -> Any:
        """Remove & returns value for `key` from metadata;
        if not found, return `default` if specified,
        otherwise raise `KeyError`"""
        if key in self.__dict__:
            return self.__dict__.pop(key)
        elif default != __DEFAULT:
            return default
        else:
            raise KeyError(f"{key} not found in metadata")

    def keys(self) -> Iterable[str]:
        """Return an iterator over the keys in metadata"""
        return self.__dict__.keys()

    def values(self) -> Iterable[Any]:
        """Return an iterator over the values in metadata"""
        return self.__dict__.values()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return an iterator over <key, value> pairs in metadata"""
        return self.__dict__.items()

    # Interfaces from loading/saving to dictionary
    def to_json(self) -> Dict[str, Any]:
        """Return a dict representation of metadata"""
        return deepcopy(self.__dict__)

    @classmethod
    def from_json(cls, di: Dict[str, Any]) -> "Metadata":
        """Create a Metadata object from a dict representation"""
        metadata = cls()
        for k, v in di.items():
            metadata.set(k, v)
        return metadata

    # The following methods are to ensure equality between metadata
    # with same keys and values
    def __len__(self) -> int:
        return len(self.__dict__)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Metadata):
            return False
        if len(self) != len(__o):
            return False
        for k in __o.keys():
            if k not in self.keys() or self[k] != __o[k]:
                return False
        return True

    # The following methods are for compatibility with the dict interface
    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __iter__(self) -> Iterable[str]:
        return self.keys()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.set(key, value)

    # The following methods are for compatibility with the dot access interface
    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        return self.set(key, value)

    def __delattr__(self, key: str) -> Any:
        return self.pop(key)

    # The following methods return a nice representation of the metadata
    def __repr__(self) -> str:
        return f"Metadata({repr(self.__dict__)})"

    def __str__(self) -> str:
        return f"Metadata({str(self.__dict__)})"

    # Finally, we need to support pickling/copying
    def __deepcopy__(self, memo: Dict[int, Any]) -> "Metadata":
        return Metadata.from_json(deepcopy(self.__dict__, memo))


T = TypeVar("T")


def store_field_in_metadata(
    field_name: str,
    getter_fn: Optional[Callable[[T], Any]] = None,
    setter_fn: Optional[Callable[[T, Any], None]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """This decorator is used to store a field that was previously part of
    a dataclass in a metadata attribute, while keeping it accessible as an
    attribute using the .field_name notation. Example:

    @store_field_in_metadata('field_a')
    @dataclass
    class MyDataclass:
        metadata: Metadata = field(default_factory=Metadata)
        field_a: int = 3
        field_b: int = 4

    d = MyDataclass()
    print(d.field_a) # 3
    print(d.metadata)   # Metadata({'field_a': 3})
    print(d) # MyDataclass(field_a=3, field_b=4, metadata={'field_a': 3})

    d = MyDataclass(field_a=5)
    print(d.field_a) # 5
    print(d.metadata)   # Metadata({'field_a': 5})
    print(d) # MyDataclass(field_a=5, field_b=4, metadata={'field_a': 5})

    Args:
        field_name: The name of the field to store in the metadata.
        getter_fn: A function that takes the dataclass instance and
            returns the value of the field. If None, the field's value is
            looked up from the metadata dictionary.
        setter_fn: A function that takes the dataclass instance and a value
            for the field and sets it. If None, the field's value is added
            to the metadata dictionary.
    """

    def wrapper(
        cls_: Type[T],
        wrapper_field_name: str = field_name,
        wrapper_getter_fn: Optional[Callable[[T], Any]] = getter_fn,
        wrapper_setter_fn: Optional[Callable[[T, Any], None]] = setter_fn,
    ) -> Type[T]:
        if not (is_dataclass(cls_)):
            raise TypeError("add_deprecated_field only works on dataclasses")

        dataclass_fields = {field.name: field for field in fields(cls_)}

        # ensures we have a metadata dict where to store the field value
        if "metadata" not in dataclass_fields:
            raise TypeError(
                "add_deprecated_field requires a `metadata` field"
                "in the dataclass of type dict."
            )

        if not issubclass(dataclass_fields["metadata"].type, Metadata):
            raise TypeError(
                "add_deprecated_field requires a `metadata` field "
                "in the dataclass of type Metadata, not "
                f"{dataclass_fields['metadata'].type}."
            )

        # ensure the field is declared in the dataclass
        if wrapper_field_name not in dataclass_fields:
            raise TypeError(
                f"add_deprecated_field requires a `{wrapper_field_name}` field"
                "in the dataclass."
            )

        if wrapper_getter_fn is None:
            # create property for the deprecated field, as well as a setter
            # that will add to the underlying metadata dict
            def _wrapper_getter_fn(
                self, field_spec: Field = dataclass_fields[wrapper_field_name]
            ):
                # we expect metadata to be of type Metadata
                metadata: Union[Metadata, None] = getattr(
                    self, "metadata", None
                )
                if metadata is not None and wrapper_field_name in metadata:
                    return metadata.get(wrapper_field_name)
                elif field_spec.default is not MISSING:
                    return field_spec.default
                elif field_spec.default_factory is not MISSING:
                    return field_spec.default_factory()
                else:
                    raise AttributeError(
                        f"Value for attribute '{wrapper_field_name}' "
                        "has not been set."
                    )

            # this avoids mypy error about redefining an argument
            wrapper_getter_fn = _wrapper_getter_fn

        field_property = property(wrapper_getter_fn)

        if wrapper_setter_fn is None:

            def _wrapper_setter_fn(self: T, value: Any) -> None:
                # need to use getattr otherwise pylance complains
                # about not knowing if 'metadata' is available as an
                # attribute of self (which it is, since we checked above
                # that it is in the dataclass fields)
                metadata: Union[Metadata, None] = getattr(
                    self, "metadata", None
                )

                if metadata is None:
                    raise RuntimeError(
                        "all deprecated fields must be declared after the "
                        f"`metadata` field; however, `{wrapper_field_name}`"
                        " was declared before. Fix your class definition."
                    )

                metadata.set(wrapper_field_name, value)

            # this avoids mypy error about redefining an argument
            wrapper_setter_fn = _wrapper_setter_fn

        # make a setter for the deprecated field
        field_property = field_property.setter(wrapper_setter_fn)

        # assign the property to the dataclass
        setattr(cls_, wrapper_field_name, field_property)

        return cls_

    return wrapper
