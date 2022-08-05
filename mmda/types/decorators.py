from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, Callable, Optional, Type, TypeVar, Union

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
        metadata: Dict[str, Any] = field(default_factory=dict)
        field_a: int = 3
        field_b: int = 4

    d = MyDataclass()
    print(d.field_a) # 3
    print(d.metadata) # {'field_a': 3}
    print(d) # MyDataclass(field_a=3, field_b=4, metadata={'field_a': 3})

    d = MyDataclass(field_a=5)
    print(d.field_a) # 5
    print(d.metadata) # {'field_a': 5}
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
                if wrapper_field_name in (
                    metadata := getattr(self, "metadata", {})
                ):
                    return metadata[wrapper_field_name]
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
                metadata: Union[dict, None] = getattr(self, "metadata", None)

                if metadata is None:
                    raise RuntimeError(
                        "all deprecated fields must be declared after the "
                        f"`metadata` field; however, `{wrapper_field_name}`"
                        " was declared before. Fix your class definition."
                    )

                metadata[wrapper_field_name] = value

            # this avoids mypy error about redefining an argument
            wrapper_setter_fn = _wrapper_setter_fn

        # make a setter for the deprecated field
        field_property = field_property.setter(wrapper_setter_fn)

        # assign the property to the dataclass
        setattr(cls_, wrapper_field_name, field_property)

        return cls_

    return wrapper
