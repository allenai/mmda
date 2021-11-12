"""

@rauthur

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from mmda.types.box import Box
from mmda.types.image import PILImage


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
                self._callback(v)
        else:
            self._callback(value)

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


class Document(UserDataMixin):
    symbols: str
    images: Optional[Iterable[PILImage]]

    def __init__(self, symbols: str):
        super().__init__(after_set_callback=self._set_callback)

        self.symbols = symbols

    def attach_images(self, images: Iterable[PILImage]) -> None:
        """Attach page images to this document. Overwrites any that are present.

        Args:
            images (Iterable[PILImage]): Iterable of individual pages in order
        """
        self.images = images

    def _set_callback(self, value: Any) -> None:
        """Overrides method in UserDataMixin to attach document to all span/box groups.

        Args:
            value (Any): A span or box group (DocAttachable)
        """
        value.attach_doc(self)


@dataclass
class Span:
    start: int
    end: int
    box: Box

    @property
    def page(self):
        return self.box.page


class DocAttachable:
    doc: Optional[Document] = None

    def attach_doc(self, doc: Document):
        if not self.doc:
            self.doc = doc
        else:
            raise AttributeError(f"This annotation already has an attached document")

    def attach_doc_callback(self, value: Any) -> None:
        """Attach document on this attachable to any child annotations.

        Args:
            value (Any): A span or box group (DocAttachable)
        """
        if self.doc and isinstance(value, DocAttachable):
            value.attach_doc(self)


@dataclass
class BoxGroup(UserDataMixin, DocAttachable):
    boxes: Iterable[Box]

    def __init__(self, boxes: Iterable[Box]):
        super().__init__(after_set_callback=self.attach_doc_callback)
        self.boxes = boxes


@dataclass
class SpanGroup(UserDataMixin, DocAttachable):
    spans: Iterable[Span]

    def __init__(self, spans: Iterable[Span]):
        super().__init__(after_set_callback=self.attach_doc_callback)
        self.spans = spans

    @property
    def symbols(self) -> Iterable[str]:
        return [self.doc.symbols[span.start : span.end] for span in self.spans]

    def __getitem__(self, key: int):
        return self.spans[key]

    @property
    def start(self) -> int:
        return (
            min([span.start for span in self.spans])
            if len(self.spans) > 0
            else float("-inf")
        )

    @property
    def end(self) -> int:
        return (
            max([span.end for span in self.spans])
            if len(self.spans) > 0
            else float("inf")
        )

    def __lt__(self, other: "SpanGroup"):
        if self.id and other.id:
            return self.id < other.id
        else:
            return self.start < other.start
