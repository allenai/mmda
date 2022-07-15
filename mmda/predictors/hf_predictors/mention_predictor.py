import os

from optimum.onnxruntime import ORTModelForTokenClassification
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, PretrainedConfig
from typing import List

from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.types.span import Span


class MentionPredictor:
    def __init__(self, artifacts_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)

        onnx_model_path = os.path.join(artifacts_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            self.model = ORTModelForTokenClassification.from_pretrained(artifacts_dir, file_name="model.onnx")
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(artifacts_dir)

    def predict(self, doc: Document) -> List[SpanGroup]:
        ret = []

        for page in doc.pages:
            words: List[str] = ["".join(token.symbols) for token in page.tokens]
            word_spans: List[List[Span]] = [token.spans for token in page.tokens]

            inputs = self.tokenizer(
                [words],
                is_split_into_words=True,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_overflowing_tokens=True,
                return_tensors="pt"
            )
            del inputs["overflow_to_sample_mapping"]
            logits = self.model(**inputs).logits

            prediction_label_ids = torch.argmax(logits, dim=-1)

            for idx1 in range(len(inputs['input_ids'])):
                label_ids = prediction_label_ids[idx1]
                input = inputs[idx1]

                word_ids: List[int] = [input.word_ids[0]] if input.word_ids[0] is not None else []
                word_label_ids: List[List[int]] = [[label_ids[0]]] if input.word_ids[0] is not None else []

                for idx2 in range(1, len(input.word_ids)):
                    word_id: int = input.word_ids[idx2]
                    previous_word_id: int = input.word_ids[idx2 - 1]

                    if word_id is not None:
                        label_id: int = label_ids[idx2]

                        if word_id == previous_word_id:
                            word_label_ids[-1].append(label_id)
                        else:
                            word_label_ids.append([label_id])
                            word_ids.append(word_id)

                acc: List[Span] = []

                for word_id, label_ids in zip(word_ids, word_label_ids):
                    spans = word_spans[word_id]

                    has_begin = any(self.model.config.id2label[int(lbl)] == "B-MENTION" for lbl in label_ids)
                    has_inside = any(self.model.config.id2label[int(lbl)] == "I-MENTION" for lbl in label_ids)

                    if has_inside:
                        acc.extend(spans)
                    elif has_begin:
                        if acc:
                            ret.append(SpanGroup(spans=acc))
                        acc = spans
                    elif acc:
                        ret.append(SpanGroup(spans=acc))
                        acc = []

                if acc:
                    ret.append(SpanGroup(spans=acc))
        return ret

