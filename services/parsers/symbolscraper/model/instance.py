from pydantic import BaseModel, Field


class Instance(BaseModel):
    """Represents one object for which inference can be performed."""

    pdf: str = Field(description="Base64 encoded bytes of PDF")
