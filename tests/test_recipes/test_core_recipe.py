"""

@kylel

"""

import os
import unittest

from mmda.recipes.recipe import CoreRecipe

FIRST_1000_SYMBOLS = """Field\nTask\nDataset\nSOTA\nB ERT -Base\nS CI B ERT\nFrozen\nFinetune\nFrozen\nFinetune\nBio\nNER\nBC5CDR (Li et al., 2016)\n88.85 7\n85.08\n86.72\n88.73\n90.01\nJNLPBA (Collier and Kim, 2004)\n78.58\n74.05\n76.09\n75.77\n77.28\nNCBI-disease (Dogan et al., 2014)\n89.36\n84.06\n86.88\n86.39\n88.57\nPICO\nEBM-NLP (Nye et al., 2018)\n66.30\n61.44\n71.53\n68.30\n72.28\nDEP\nGENIA (Kim et al., 2003) - LAS\n91.92\n90.22\n90.33\n90.36\n90.43\nGENIA (Kim et al., 2003) - UAS\n92.84\n91.84\n91.89\n92.00\n91.99\nREL\nChemProt (Kringelum et al., 2016)\n76.68\n68.21\n79.14\n75.03\n83.64\nCS\nNER\nSciERC (Luan et al., 2018)\n64.20\n63.58\n65.24\n65.77\n67.57\nREL\nSciERC (Luan et al., 2018)\nn/a\n72.74\n78.71\n75.25\n79.97\nCLS\nACL-ARC (Jurgens et al., 2018)\n67.9\n62.04\n63.91\n60.74\n70.98\nMulti\nCLS\nPaper Field\nn/a\n63.64\n65.37\n64.38\n65.71\nSciCite (Cohan et al., 2019)\n84.0\n84.31\n84.85\n85.42\n85.49\nAverage\n73.58\n77.16\n76.01\n79.27\nTable 1: Test performances of all B ERT variants on all tasks and datasets. Bold indicates the SOTA result (multiple\nresults bolded if difference wi"""
PAGE_JSON = {'spans': [{'start': 0, 'end': 3696,
                        'box': {'left': 0.12100741176470588, 'top': 0.08015236441805225,
                                'width': 0.7625643173109246, 'height': 0.8289201816627079,
                                'page': 0}}], 'id': 0, 'metadata': {}}


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.pdfpath = os.path.join(os.path.dirname(__name__), '../fixtures/1903.10676.pdf')

    def test_run(self):
        recipe = CoreRecipe()
        doc = recipe.run(pdfpath=self.pdfpath)
        self.assertEqual(doc.symbols[:1000], FIRST_1000_SYMBOLS)
        self.assertDictEqual(doc.pages[0].to_json(), PAGE_JSON)

