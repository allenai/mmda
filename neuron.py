#!/usr/bin/env python3
from typing import *
import json
import sys
from timeit import default_timer as timer

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document
from mmda.types.api import SpanGroup


input_file = "/home/yogic/data/0193f03c92707c675163ab0939ec931b16278502-34-request.json"
output_file = "/home/yogic/weights/trace.pt"
artifacts_dir = "/home/yogic/weights"


def mk_inputs():
    with open(input_file) as f:
        inst = json.load(f)['instances'][0]
    doc = Document(inst['symbols'])
    doc.annotate(tokens=[SpanGroup(**t).to_mmda() for t in inst['tokens']])
    doc.annotate(pages=[SpanGroup(**p).to_mmda() for p in inst['pages']])

    tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)

    page = doc.pages[0]
    words = ["".join(token.symbols) for token in page.tokens]
    inputs = tokenizer(
        [words],
        is_split_into_words=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_overflowing_tokens=True,
        return_tensors="pt"
    )
    del inputs['overflow_to_sample_mapping']
    
    return doc, inputs


def gen():
    model = AutoModelForTokenClassification.from_pretrained(artifacts_dir, return_dict=False)
    _, inputs = mk_inputs()

    # convert to tuple for neuron model
    neuron_inputs = (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
    model_neuron = torch.jit.trace(model, neuron_inputs)
    model.config.update({"traced_sequence_length": 512})
    model_neuron.save(output_file)


def predict():
    model1 = MentionPredictor(artifacts_dir)
    doc1, _ = mk_inputs()
    s1 = timer()
    pred1 = model1.predict(doc1)
    e1 = timer()

    model2 = MentionPredictor(artifacts_dir, torchscript=True)
    doc2, _ = mk_inputs()
    s2 = timer()
    pred2 = model2.predict(doc2)
    e2 = timer()

    print(f"Regular model took {e1 - s1}")
    for sg in pred1:
        print(json.dumps(SpanGroup.from_mmda(sg).dict()))

    print(f"\nTorchscript model took {e2 - s2}")
    for sg in pred2:
        print(json.dumps(SpanGroup.from_mmda(sg).dict()))


def main():
    cmd = "pred"
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    if cmd == "gen": 
        gen()
    elif cmd == "pred": 
        predict()


if __name__ == "__main__":
    main()
