import numpy as np
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Tuple, Dict
import xgboost as xgb

from mmda.types import api
from mmda.featurizers.citation_link_featurizers import CitationLink, featurize

class CitationLinkPredictor:
    def __init__(self, artifacts_dir: str):
        full_model_path = os.path.join(artifacts_dir, "links_v0.json")
        model = xgb.XGBClassifier()
        model.load_model(full_model_path)
        self.model = model
    
    # returns linked mentions: the mention SpanGroup + the id for the linked bibs SpanGroup
    def predict(self, mentions: List[api.SpanGroup], bibs: List[api.SpanGroup]) -> List[Tuple[api.SpanGroup, str]]:
        predicted_links = []
        
        # iterate over mentions
        for mention in mentions:
            # create possible links
            possible_links = [] 
            for bib in bibs:
                link = CitationLink(mention = mention, bib = bib)
                possible_links.append(link)
            
            # featurize and predict
            X_instances = featurize(possible_links)
            y_pred = self.model.predict_proba(X_instances)
            match_scores = [pred[1] for pred in y_pred] # probability that label is 1
            match_index = np.argmax(match_scores)
            selected_link = possible_links[match_index]
            predicted_links.append((selected_link.mention, selected_link.bib.id))
        
        return predicted_links

