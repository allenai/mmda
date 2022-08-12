import pathlib
import datetime, json
import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import transformers
import os
import warnings
from pathlib import Path

from mmda.parsers.pdfplumber_parser import PDFPlumberParser, _SPLIT_AT_PUNCTUATION
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document

from interface import Instance

num_cores = 16 # This value should be 4 on inf1.xlarge and inf1.2xlarge
nc_env = ','.join(['1'] * num_cores)
os.environ['NEURONCORE_GROUP_SIZES'] = nc_env


# download your pdfs here
test_dir = pathlib.Path.home() / "test_data_requests" 
# download from s3://ai2-s2-mmda/models/citation-mentions/2022-07-27-minilm-10k/model/artifacts.tar.gz
artifacts_dir = pathlib.Path.home() / "fangzhou" / "weights"

# Loop through pdf
#file = pdfs_dir / "382098006be8d6e8541bf74956d80eb9781f738e-294-request.json"

# Loop through instances
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
 
def test_bench(file_list):
    predictor = MentionPredictor(artifacts_dir).predict
    # save the mention list?
    [add_mentions_to_doc(file_path, predictor) for file_path in file_list]


file_list = list(test_dir.glob('**/*.json'))[:1000]
len_requests = len(file_list)
print(f"There are {len_requests} requests")

# time cost calc, there must be a beter way
start = datetime.datetime.now()

test_bench(file_list)

end = datetime.datetime.now()
total_time = (end - start).seconds
average = "{:.2f}".format(total_time / len_requests)
# p50
print("\n"+ average + " s per each request")
