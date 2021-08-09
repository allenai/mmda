from dataclasses import dataclass
from typing import Union, List, Dict, Any

# TODO[Shannon] Not exactly sure about the model configurations till
# we add specific models.
@dataclass
class BaseDocumentPredictorConfig:
    model_name: str = None
    model_class: Any = None