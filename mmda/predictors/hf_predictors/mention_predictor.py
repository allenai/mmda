import itertools
import os.path
from typing import Dict, Iterator, List, Optional

from optimum.onnxruntime import ORTModelForTokenClassification
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch.neuron

from mmda.types.annotation import Annotation, SpanGroup
from mmda.types.document import Document
from mmda.types.span import Span

# set this to true to procdude the neuron traced classifer 
NEURON_KICKOFF = True
NEURON = True

class Labels:
    # BILOU https://stackoverflow.com/q/17116446

    MENTION_OUTSIDE_ID = 0
    MENTION_BEGIN_ID = 1
    MENTION_INSIDE_ID = 2
    MENTION_LAST_ID = 3
    MENTION_UNIT_ID = 4

    MENTION_OUTSIDE = "O"
    MENTION_BEGIN = "B-MENTION"
    MENTION_INSIDE = "I-MENTION"
    MENTION_LAST = "E-MENTION"  # "end"
    MENTION_UNIT = "S-MENTION"  # "single"

    ID_TO_LABEL: Dict[int, str] = {
        MENTION_OUTSIDE_ID: MENTION_OUTSIDE,
        MENTION_BEGIN_ID: MENTION_BEGIN,
        MENTION_INSIDE_ID: MENTION_INSIDE,
        MENTION_LAST_ID: MENTION_LAST,
        MENTION_UNIT_ID: MENTION_UNIT
    }
    LABEL_TO_ID: Dict[str, int] = {
        MENTION_OUTSIDE: MENTION_OUTSIDE_ID,
        MENTION_BEGIN: MENTION_BEGIN_ID,
        MENTION_INSIDE: MENTION_INSIDE_ID,
        MENTION_LAST: MENTION_LAST_ID,
        MENTION_UNIT: MENTION_UNIT_ID
    }


class MentionPredictor:
    def __init__(self, artifacts_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)

        onnx = os.path.exists(os.path.join(artifacts_dir, "model.onnx"))
        if onnx:
            self.model = ORTModelForTokenClassification.from_pretrained(artifacts_dir, file_name="model.onnx")
        else:
            if NEURON or NEURON_KICKOFF:
                #Encountering a dict at the output of the tracer might cause the trace to be incorrect
                self.model = AutoModelForTokenClassification.from_pretrained(artifacts_dir, return_dict=False)
            else:
                self.model = AutoModelForTokenClassification.from_pretrained(artifacts_dir)
        
        # this is a side-effect(y) function
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if not onnx:
            # https://stackoverflow.com/a/60018731
            self.model.eval()  # for some reason the onnx version doesnt have an eval()

        if NEURON and not NEURON_KICKOFF:
            # location to the neuron classifer
            self.model = torch.jit.load(os.path.join("/home/ubuntu/fangzhou/model/", 'hack_neuron_model.pt'))

    def predict(self, doc: Document, print_warnings: bool = False) -> List[SpanGroup]:
        if not hasattr(doc, 'pages'):
            return []

        spangroups = []
        for page in doc.pages:
            spangroups.extend(self.predict_page(page, counter=itertools.count(), print_warnings=print_warnings))
        return spangroups

    def predict_page(self, page: Annotation, counter: Iterator[int], print_warnings: bool = False) -> List[SpanGroup]:
        if not hasattr(page, 'tokens'):
            return []

        ret = []
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

        if not NEURON:
            inputs = inputs.to(self.model.device)

        del inputs["overflow_to_sample_mapping"]

        # convert to tuple for neuron model
        neuron_inputs = (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

        if NEURON_KICKOFF:            
            # save Neuron model here 
            model_neuron = torch.neuron.trace(self.model, neuron_inputs)
            self.model.config.update({"traced_sequence_length": 512})
            save_dir = "/home/ubuntu/fangzhou/model/"
            model_neuron.save(os.path.join(save_dir,"hack_neuron_model.pt"))
            # replace the original ones:
            self.model = model_neuron

        with torch.no_grad():
            if not NEURON: 
                outputs = self.model(**inputs)
                print(outputs)
                
                prediction_label_ids = torch.argmax(outputs.logits, dim=-1)
            else:
                # call the model differently 
                outputs = self.model(*neuron_inputs)
                print(*neuron_inputs)
                print(outputs)
                prediction_label_ids = torch.argmax(outputs[0], dim=-1)


        def has_label_id(lbls: List[int], want_label_id: int) -> bool:
            return any(lbl == want_label_id for lbl in lbls)

        for idx1 in range(len(inputs['input_ids'])):
            batch_label_ids = prediction_label_ids[idx1]
            input_ = inputs[idx1]

            word_ids: List[int] = [input_.word_ids[0]] if input_.word_ids[0] is not None else []
            word_label_ids: List[List[int]] = [[batch_label_ids[0]]] if input_.word_ids[0] is not None else []

            for idx2 in range(1, len(input_.word_ids)):
                word_id: int = input_.word_ids[idx2]
                previous_word_id: int = input_.word_ids[idx2 - 1]

                if word_id is not None:
                    label_id: int = batch_label_ids[idx2]
                    if word_id == previous_word_id:
                        word_label_ids[-1].append(label_id)
                    else:
                        word_label_ids.append([label_id])
                        word_ids.append(word_id)

            acc: List[Span] = []
            outside_mention = True

            def append_acc():
                nonlocal acc
                if acc:
                    ret.append(SpanGroup(spans=acc, id=next(counter)))
                acc = []

            for word_id, label_ids in zip(word_ids, word_label_ids):
                spans = word_spans[word_id]

                has_begin = has_label_id(label_ids, Labels.MENTION_BEGIN_ID)
                has_last = has_label_id(label_ids, Labels.MENTION_LAST_ID)
                has_unit = has_label_id(label_ids, Labels.MENTION_UNIT_ID)

                warnings = []
                label_id: Optional[int] = None

                if sum(1 for cond in [has_begin, has_last, has_unit] if cond) > 1:
                    warnings.append(
                        "found multiple labels for the same word: "
                        f"has_begin={has_begin} has_last={has_last} has_unit={has_unit}"
                    )
                    for cur_label_id in label_ids:
                        # prioritize begin, last, unit over the rest
                        if cur_label_id not in (Labels.MENTION_INSIDE_ID, Labels.MENTION_OUTSIDE_ID):
                            label_id = cur_label_id
                            break

                if label_id is None:
                    # prioritize inside over outside
                    label_id = Labels.MENTION_INSIDE_ID \
                        if any(lbl == Labels.MENTION_INSIDE_ID for lbl in label_ids) else label_ids[0]

                if outside_mention and has_last:
                    warnings.append('found an "L" while outside mention')
                if not outside_mention and (has_begin or has_unit):
                    warnings.append('found an "L" or "U" while inside mention')

                if warnings and print_warnings:
                    print("warnings:")
                    for warning in warnings:
                        print(f"  - {warning}")

                if label_id == Labels.MENTION_UNIT_ID:
                    append_acc()
                    acc = spans
                    append_acc()
                    outside_mention = True
                if label_id == Labels.MENTION_BEGIN_ID:
                    append_acc()
                    acc = spans
                    outside_mention = False
                elif label_id == Labels.MENTION_LAST_ID:
                    acc.extend(spans)
                    append_acc()
                    outside_mention = True
                elif label_id == Labels.MENTION_INSIDE_ID:
                    acc.extend(spans)
            append_acc()
        return ret
