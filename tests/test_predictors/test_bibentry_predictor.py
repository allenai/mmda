import unittest

from mmda.predictors.hf_predictors.bibentry_predictor.types import (
    BibEntryPredictionWithSpan,
    StringWithSpan
)
from mmda.predictors.hf_predictors.bibentry_predictor import utils
from mmda.types.annotation import SpanGroup
from mmda.types.span import Span


class TestBibEntryPredictor(unittest.TestCase):
    def test__map_raw_predictions_to_mmda(self):
        sg = SpanGroup(spans=[Span(start=17778, end=17832, box=None), Span(start=20057, end=20233, box=None)])
        raw_prediction = BibEntryPredictionWithSpan(
            citation_number=StringWithSpan(content='10', start=0, end=2),
            authors=[
                StringWithSpan(content='Srivastava, K.', start=4, end=18),
                StringWithSpan(content='V.B. Upadhyay', start=23, end=36)
            ],
            title=StringWithSpan(content='Effect of Phytoecdysteroid on Length of Silk Filament and\nNon-Breakable Filament Length of Multivoltine\nMulberry Silkworm B. mori Linn', start=45, end=179),
            journal_venue_or_event=StringWithSpan(content='Academic Journal\nof Entomology', start=181, end=211),
            year=None,
            doi=None,
            url=None
        )

        # Real world example
        total_doc_text = "." * sg.spans[0].start \
                         + "10. Srivastava, K. and V.B. Upadhyay, 2012b. Effect of" \
                         + "." * (sg.spans[1].start - sg.spans[0].end) \
                         + "Phytoecdysteroid on Length of Silk Filament and\nNon-Breakable Filament Length of Multivoltine\nMulberry Silkworm B. mori Linn. Academic Journal\nof Entomology, 5 ( 3 ) : 174-181."

        prediction = utils.map_raw_predictions_to_mmda([sg], [raw_prediction])[0]

        self.assertEqual(total_doc_text[prediction.bib_entry_number[0].start:prediction.bib_entry_number[0].end], "10")
        self.assertEqual(total_doc_text[prediction.bib_entry_authors[0].start:prediction.bib_entry_authors[0].end], "Srivastava, K.")
        self.assertEqual(total_doc_text[prediction.bib_entry_authors[1].start:prediction.bib_entry_authors[1].end], "V.B. Upadhyay")

        self.assertEqual(len(prediction.bib_entry_title[0].spans), 2)
        self.assertEqual(total_doc_text[prediction.bib_entry_title[0].spans[0].start:prediction.bib_entry_title[0].spans[0].end], "Effect of")
        self.assertEqual(total_doc_text[prediction.bib_entry_title[0].spans[1].start:prediction.bib_entry_title[0].spans[1].end], "Phytoecdysteroid on Length of Silk Filament and\nNon-Breakable Filament Length of Multivoltine\nMulberry Silkworm B. mori Linn")

        self.assertEqual(total_doc_text[prediction.bib_entry_venue_or_event[0].start:prediction.bib_entry_venue_or_event[0].end], "Academic Journal\nof Entomology")

        self.assertEqual(prediction.bib_entry_year, [])
        self.assertEqual(prediction.bib_entry_doi, [])
        self.assertEqual(prediction.bib_entry_url, [])

