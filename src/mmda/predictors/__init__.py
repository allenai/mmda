# flake8: noqa
from necessary import necessary

with necessary(["tokenizers"], soft=True) as TOKENIZERS_AVAILABLE:
    if TOKENIZERS_AVAILABLE:
        from mmda.predictors.heuristic_predictors.whitespace_predictor import WhitespacePredictor
        from mmda.predictors.heuristic_predictors.dictionary_word_predictor import DictionaryWordPredictor
        __all__ = ['DictionaryWordPredictor', 'WhitespacePredictor']


with necessary('pysbd', soft=True) as PYSBD_AVAILABLE:
    if PYSBD_AVAILABLE:
        from mmda.predictors.heuristic_predictors.sentence_boundary_predictor \
            import PysbdSentenceBoundaryPredictor
        __all__.append('PysbdSentenceBoundaryPredictor')

with necessary(["layoutparser", "torch", "torchvision", "effdet"], soft=True) as PYTORCH_AVAILABLE:
    if PYTORCH_AVAILABLE:
        from mmda.predictors.lp_predictors import LayoutParserPredictor
        __all__.append('LayoutParserPredictor')

