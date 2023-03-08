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
FIRST_10_TOKENS_JSON = [{'spans': [{'start': 0, 'end': 5,
                                    'box': {'left': 0.14541159663865547, 'top': 0.08015236441805225,
                                            'width': 0.031124640759663848,
                                            'height': 0.010648907363420378, 'page': 0}}], 'id': 0,
                         'metadata': {'fontname': 'HXONRZ+NimbusRomNo9L-Regu',
                                      'size': 8.966379999999958}}, {'spans': [
    {'start': 6, 'end': 10,
     'box': {'left': 0.2218368002857143, 'top': 0.08015236441805225, 'width': 0.028109224561344556,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 1, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 11, 'end': 18,
     'box': {'left': 0.28294983802016804, 'top': 0.08015236441805225, 'width': 0.04515740219831938,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 2, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 19, 'end': 23,
     'box': {'left': 0.5239827089210084, 'top': 0.08015236441805225, 'width': 0.03749755185546227,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 3, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 24, 'end': 25,
     'box': {'left': 0.6157472036638656, 'top': 0.08015236441805225, 'width': 0.010051387327731112,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 4, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 26, 'end': 29,
     'box': {'left': 0.6266233613445378, 'top': 0.08181785724465564, 'width': 0.02369895794957974,
             'height': 0.00851912114014249, 'page': 0}}], 'id': 5, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 7.173099999999977}}, {'spans': [
    {'start': 30, 'end': 31,
     'box': {'left': 0.6508250420168067, 'top': 0.08015236441805225, 'width': 0.005018158890756309,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 6, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 31, 'end': 35,
     'box': {'left': 0.6558673121815126, 'top': 0.08015236441805225, 'width': 0.02927711439327727,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 7, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 36, 'end': 37,
     'box': {'left': 0.7629575354285715, 'top': 0.08015236441805225, 'width': 0.008378667697478945,
             'height': 0.010648907363420378, 'page': 0}}], 'id': 8, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 8.966379999999958}}, {'spans': [
    {'start': 38, 'end': 40,
     'box': {'left': 0.7722364705882353, 'top': 0.08181785724465564, 'width': 0.012888674302521032,
             'height': 0.00851912114014249, 'page': 0}}], 'id': 9, 'metadata': {
    'fontname': 'HXONRZ+NimbusRomNo9L-Regu', 'size': 7.173099999999977}}]


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.pdfpath = os.path.join(os.path.dirname(__name__), '../fixtures/1903.10676.pdf')

    def test_run(self):
        recipe = CoreRecipe()
        doc = recipe.run(pdfpath=self.pdfpath)
        self.assertEqual(doc.symbols[:1000], FIRST_1000_SYMBOLS)
        self.assertDictEqual(doc.pages[0].to_json(), PAGE_JSON)
        self.assertListEqual([t.to_json() for t in doc.tokens[:10]], FIRST_10_TOKENS_JSON)
