"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings, Field

from ai2_internal import api
from mmda.predictors.d2_predictors.bibentry_detection_predictor import BibEntryDetectionPredictor
from mmda.types import image
from mmda.types.document import Document
from mmda.utils.tools import box_groups_to_span_groups


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    symbols: str
    tokens: List[api.SpanGroup]
    rows: List[api.SpanGroup]
    pages: List[api.SpanGroup]
    page_images: List[str] = Field(description="List of base64-encoded page images")


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """
    bib_entries: List[api.SpanGroup]
    raw_bib_entry_boxes: List[api.SpanGroup]


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    These serve as a record of the ENV
    vars the consuming application needs to set.
    """

    BIB_ENTRY_DETECTION_PREDICTOR_SCORE_THRESHOLD: float = Field(default=0.6, description="Prediction accuracy score used to determine threshold of returned predictions")


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        self._predictor = BibEntryDetectionPredictor(self._artifacts_dir, self._config.BIB_ENTRY_DETECTION_PREDICTOR_SCORE_THRESHOLD)

    def predict_one(self, inst: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        doc = Document(symbols=inst.symbols)
        doc.annotate(tokens=[sg.to_mmda() for sg in inst.tokens])
        doc.annotate(rows=[sg.to_mmda() for sg in inst.rows])
        doc.annotate(pages=[sg.to_mmda() for sg in inst.pages])
        images = [image.frombase64(im) for im in inst.page_images]
        doc.annotate_images(images)

        processed_bib_entry_box_groups, original_box_groups = self._predictor.predict(
            doc
        )

        # generate SpanGroups
        if len(processed_bib_entry_box_groups) > 0:
            bib_entry_span_groups = box_groups_to_span_groups(processed_bib_entry_box_groups, doc, pad_x=True, center=True)
            doc.annotate(bib_entries=bib_entry_span_groups)
            # remove boxes from spans to create entity-like SpanGroups with BoxGroups
            # (where the only set of boxes is on SpanGroup.box_group)
            no_span_box_span_groups = [
                api.SpanGroup(
                    # omit s.box when generating list of spans
                    spans=[api.Span(start=s.start, end=s.end) for s in sg.spans],
                    box_group=api.BoxGroup.from_mmda(sg.box_group),
                    id=sg.id
                )
                for sg in doc.bib_entries
            ]
            prediction = Prediction(
                # filter out span-less SpanGroups which occasionally occur
                bib_entries=[sg for sg in no_span_box_span_groups if len(sg.spans) != 0],
                # retain the original model output
                raw_bib_entry_boxes=[api.SpanGroup(spans=[], box_group=api.BoxGroup.from_mmda(bg), id=bg.id) for bg in original_box_groups]
            )
        else:
            prediction = Prediction(
                bib_entries=[],
                raw_bib_entry_boxes=[]
            )

        return prediction

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        return [self.predict_one(instance) for instance in instances]
