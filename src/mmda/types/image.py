"""


Monkey patch the PIL.Image methods to add base64 conversion

"""

import base64
from io import BytesIO

from PIL import Image as pilimage
from PIL.Image import Image as PILImage


def tobase64(self):
    # Ref: https://stackoverflow.com/a/31826470
    buffered = BytesIO()
    self.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def frombase64(img_str):
    # Use the same naming style as the original Image methods
    buffered = BytesIO(base64.b64decode(img_str))
    img = pilimage.open(buffered)
    return img


PILImage.tobase64 = tobase64 # This is the method applied to individual Image classes
PILImage.to_json = tobase64 # Use the same API as the others
PILImage.frombase64 = frombase64 # This is bind to the module, used for loading the images
