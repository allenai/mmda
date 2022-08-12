import pathlib
import datetime, json
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import transformers
import os
import warnings
from pathlib import Path
from timeit import default_timer as timer

from mmda.parsers.pdfplumber_parser import PDFPlumberParser, _SPLIT_AT_PUNCTUATION
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document

from interface import Instance

# Guess this is not needed as we are not using neuron trace
# num_cores = 16 # This value should be 4 on inf1.xlarge and inf1.2xlarge
# nc_env = ','.join(['1'] * num_cores)
# os.environ['NEURONCORE_GROUP_SIZES'] = nc_env


# test location
test_dir = pathlib.Path.home() / "test_data_requests" 
# model location
artifacts_dir = pathlib.Path.home() / "fangzhou" / "weights"
# number of test cases you want to run
num_req = 100


predictor_org = MentionPredictor(artifacts_dir)
predictor_torchscript = MentionPredictor(artifacts_dir, torchscript=True)

def add_mentions_to_doc(file_path, predictor:callable, verbose:bool=False):
    instance_dict = json.load(open(file_path))
    for instance in instance_dict['instances']:
        instance = Instance(**instance)
        doc = Document(symbols=instance.symbols)
        doc.annotate(tokens=[t.to_mmda() for t in instance.tokens])
        doc.annotate(pages=[p.to_mmda() for p in instance.pages])

        mentions = predictor(doc)
        if verbose: 
            print(f"mentions from request {file_path}, get mentions: {mentions}")

        return mentions
 
def test_with_jit_traced_torchscript(file_list):
    print("=====Testing model with jit compilation torchscript")
    # save the mention list somewhere?
    [add_mentions_to_doc(file_path, predictor_org.predict) for file_path in file_list]

def test_with_original_model(file_list):
    print("=====Testing original model")
    # save the mention list somewhere?
    [add_mentions_to_doc(file_path, predictor_torchscript.predict) for file_path in file_list]

def test_with_onnx_runtime_accelerator(file_list):
    # filling the model piece with correct artifact path
    pass

def test_main(test_func: callable):
    file_list = list(test_dir.glob('**/*.json'))[:num_req]
    len_requests = len(file_list)
    print(f"There are {len_requests} requests")

    # time cost calc, there must be a beter way
    start = timer()

    test_func(file_list)

    end = timer()
    total_time = (end - start)
    average = "{:.4f}".format(total_time / len_requests)
    # p50
    print("\n"+ average + " s per each request")


test_main(test_with_original_model)
test_main(test_with_jit_traced_torchscript)
