#!/usr/bin/env python3
from typing import *
import os
import pathlib
import json
import random
import sys
from timeit import default_timer as timer

from numpy import percentile
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document
from mmda.types.api import SpanGroup


input_dir = pathlib.Path("/home/yogic/data")
output_file = "/home/yogic/weights/trace.pt"
artifacts_dir = "/home/yogic/weights"

def mk_inputs(input_file):
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
    input_file = input_dir / "0193f03c92707c675163ab0939ec931b16278502-34-request.json"
    model = AutoModelForTokenClassification.from_pretrained(artifacts_dir, return_dict=False)
    _, inputs = mk_inputs(input_file)

    # convert to tuple for neuron model
    neuron_inputs = (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
    model_neuron = torch.jit.trace(model, neuron_inputs)
    model.config.update({"traced_sequence_length": 512})
    model_neuron.save(output_file)


def predict():
    files = [f for f in os.listdir(input_dir) if f.endswith("-request.json")]
    random.seed(1337)
    files_sample = random.sample(files, 1000)

    ts1 = []
    ts2 = []
    ts3 = []

    model1 = MentionPredictor(artifacts_dir)
    model2 = MentionPredictor(artifacts_dir, torchscript=True)
    model3 = MentionPredictor(artifacts_dir, onnx=True)

    for file in files_sample:
        try:
            doc, _ = mk_inputs(input_dir / file)
        except Exception:
            print(f"failed {file}.")
            continue

        s1 = timer()
        model1.predict(doc)
        e1 = timer()
        ts1.append(e1 - s1)

        s2 = timer()
        model2.predict(doc)
        e2 = timer()
        ts2.append(e2 - s2)

        s3 = timer()
        model3.predict(doc)
        e3 = timer()
        ts3.append(e3 - s3)

        print(f"finished {file}. t1={e1 - s1}, t2={e2 - s2}, t3={e3 - s3}.")

    percentiles = [50, 75, 90, 95]

    latencies = percentile(ts1, percentiles)
    print("times for regular model")
    for p, l in zip(percentiles, latencies):
        print(f"p{p} = {l}")

    latencies = percentile(ts2, percentiles)
    print("times for torchscript model")
    for p, l in zip(percentiles, latencies):
        print(f"p{p} = {l}")

    latencies = percentile(ts3, percentiles)
    print("times for onnx model")
    for p, l in zip(percentiles, latencies):
        print(f"p{p} = {l}")


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
