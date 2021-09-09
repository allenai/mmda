import json
import os
from typing import List

from pydantic import BaseModel, BaseSettings, Field

from model.instance import Instance
from model.prediction import Prediction, Box, BoxGroup
import mmda.predictors.lp_predictors as lpp
from mmda.types.annotation import BoxGroup as AnnotationBox
from mmda.types.document import Document
from mmda.types.image import Image


class PredictorConfig(BaseSettings):
    """
    Path to the model weights
    """

    weights_path = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config"


class Predictor:
    """
    Initialize model
    """

    _lp_predictor: lpp.LayoutParserPredictor

    def __init__(self, config: PredictorConfig):
        """
        Load weights
        """
        self._lp_predictor = lpp.LayoutParserPredictor.from_pretrained(
            config.weights_path
        )

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        predictions = []

        for instance in instances:
            doc = Document(
                symbols="", images=[Image.frombase64(i) for i in instance.page_images]
            )
            annotations = self._lp_predictor.predict(doc)

            predictions.append(
                Prediction(
                    groups=[
                        BoxGroup(
                            boxes=[
                                Box(
                                    left=b.l,
                                    top=b.t,
                                    width=b.w,
                                    height=b.h,
                                    page=b.page,
                                )
                                for b in bg.boxes
                            ],
                            type=bg.type,
                        )
                        for bg in annotations
                    ]
                )
            )

        return predictions
