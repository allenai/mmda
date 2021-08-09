from typing import List, Union, Dict, Any, Tuple

import torch
import torch.nn.parallel
from effdet import create_model
from effdet.data.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    transforms_coco_eval,
)

from mmda.types.document import Document
from mmda.types.annotation import BoxGroup
from mmda.types.box import Box
from mmda.types.image import Image
from mmda.types.names import *
from mmda.models.base_document_predictor import BaseDocumentPredictor
from mmda.models.base_predictor_config import BasePredictorConfig


class InputTransform:
    def __init__(
        self,
        image_size,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):

        self.mean = mean
        self.std = std

        self.transform = transforms_coco_eval(
            image_size,
            interpolation="bilinear",
            use_prefetcher=True,
            fill_color="mean",
            mean=self.mean,
            std=self.std,
        )

        self.mean_tensor = torch.tensor([x * 255 for x in mean]).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([x * 255 for x in std]).view(1, 3, 1, 1)

    def preprocess(self, image: Image) -> Tuple[torch.Tensor, Dict]:

        image = image.convert("RGB")
        image_info = {"img_size": image.size}

        input, image_info = self.transform(image, image_info)
        image_info = {
            key: torch.tensor(val).unsqueeze(0) for key, val in image_info.items()
        }

        input = torch.tensor(input).unsqueeze(0)
        input = input.float().sub_(self.mean_tensor).div_(self.std_tensor)

        return input, image_info


class EfficientDetLayoutModel(BaseDocumentPredictor):

    REQUIRED_BACKENDS = ["effdet", "torch"]
    REQUIRED_DOCUMENT_FIELDS = [Images]

    def __init__(
        self,
        model: Any,
        config: BasePredictorConfig,
    ):
        super().__init__(model, config)

        self.device = config.device
        self.label_map = config.label_map if config.label_map is not None else {}
        self.output_confidence_threshold = config.output_confidence_threshold

    @classmethod
    def from_pretrained(cls, model_path:str, config: BasePredictorConfig):

        model = create_model(
            config.model_name,
            num_classes=config.num_classes,
            bench_task="predict",
            pretrained=True,
            checkpoint_path=model_path,
        )
        model.eval()
        return cls(model, config)

    def initialize_preprocessor(self) -> None:
        self.preprocessor = InputTransform(self.config.image_size)

    def preprocess(self, image: Image) -> Dict:

        image, image_info = self.preprocessor.preprocess(image)
        image = image.to(self.device)
        image_info = {key: val.to(self.device) for key, val in image_info.items()}
        return image, image_info

    def predict(self, document: Document) -> List[BoxGroup]:

        document_prediction = []

        for image_index, image in enumerate(document.images):
            image, image_info = self.preprocess(image)
            model_outputs = self.model(image, image_info)
            document_prediction.extend(self.postprocess(model_outputs, image_index))
        
        return document_prediction

    def postprocess(
        self, model_outputs: torch.Tensor, image_index: int
    ) -> List[BoxGroup]:

        model_outputs = model_outputs.cpu().detach()
        box_predictions = []

        for index, sample in enumerate(model_outputs):
            sample[:, 2] -= sample[:, 0]
            sample[:, 3] -= sample[:, 1]
            for det in sample:
                score = float(det[4])
                if (
                    score < self.output_confidence_threshold
                ):  # stop when below this threshold, scores in descending order
                    break
                box_predictions.append(
                    BoxGroup(
                        boxes=[Box(*det[0:4].tolist(), page=image_index)],
                        id=index,
                        type=self.label_map.get(int(det[5]), int(det[5])),
                    )
                )
        return box_predictions