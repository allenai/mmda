import numpy as np
import os
import pandas as pd
from typing import List, Dict, Tuple
import xgboost as xgb

from mmda.types.document import Document
from mmda.featurizers.citation_link_featurizers import CitationLink, featurize

class CitationLinkPredictor:
    def __init__(self, artifacts_dir: str):
        full_model_path = os.path.join(artifacts_dir, "links_v0.json")
        model = xgb.XGBClassifier()
        model.load_model(full_model_path)
        self.model = model

    # returns a paired mention id and bib id to represent a link
    def predict(self, doc: Document) -> List[Tuple[str, str]]:
        if len(doc.bibs) == 0:
            return []

        predicted_links = []

        # iterate over mentions
        for mention in doc.mentions:
            # create all possible links for this mention
            possible_links = []
            for bib in doc.bibs:
                link = CitationLink(mention = mention, bib = bib)
                possible_links.append(link)

            # featurize and find link with highest score
            X_instances = featurize(possible_links)
            y_pred = self.model.predict_proba(X_instances)
            match_scores = [pred[1] for pred in y_pred] # probability that label is 1
            match_index = np.argmax(match_scores)
            selected_link = possible_links[match_index]
            predicted_links.append((selected_link.mention.id, selected_link.bib.id))

        return predicted_links


