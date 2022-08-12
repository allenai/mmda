import pathlib
import time, json
import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import transformers
import os
import warnings


from mmda.parsers.pdfplumber_parser import PDFPlumberParser, _SPLIT_AT_PUNCTUATION
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document

from interface import Instance

num_cores = 16 # This value should be 4 on inf1.xlarge and inf1.2xlarge
nc_env = ','.join(['1'] * num_cores)
os.environ['NEURONCORE_GROUP_SIZES'] = nc_env


# download your pdfs here
pdfs_dir = pathlib.Path.home() / "test_data_requests" 
# download from s3://ai2-s2-mmda/models/citation-mentions/2022-07-27-minilm-10k/model/artifacts.tar.gz
artifacts_dir = pathlib.Path.home() / "fangzhou" / "weights"

# Loop through pdf
file = pdfs_dir / "382098006be8d6e8541bf74956d80eb9781f738e-294-request.json"

# Loop through instances
def add_mentions_to_doc(instance_dict:dict, predictor:callable):
    for instance in instance_dict['instances']:
        instance = Instance(**instance)
        doc = Document(symbols=instance.symbols)
        doc.annotate(tokens=[t.to_mmda() for t in instance.tokens])
        doc.annotate(pages=[p.to_mmda() for p in instance.pages])

        mentions = predictor(doc)
        print(mentions)
        doc.annotate(mentions=mentions)    

        print(f"mentions from request {file}")
        for mention in doc.mentions:
            print(mention.symbols)

instance_dict = json.load(open(file))
model = MentionPredictor(artifacts_dir)
predictor = MentionPredictor(artifacts_dir).predict

#Example from AWS Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html
# Setting up NeuronCore groups for inf1.6xlarge with 16 cores
add_mentions_to_doc(instance_dict, predictor)
