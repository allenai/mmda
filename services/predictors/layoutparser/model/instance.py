from pydantic import BaseModel, Field
from typing import List


class Instance(BaseModel):
    """Input is a list of page images, base64-encoded"""

    page_images: List[str] = Field(description="List of base64-encoded page images")
