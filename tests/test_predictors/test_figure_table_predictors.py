import pickle
import unittest
from collections import defaultdict
import pathlib
import pytest

from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions
from mmda.types import Document
from mmda.types.box import Box
from mmda.types.span import Span
import mmda.types.annotation as mmda_ann


class TestFigureCaptionPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.fixture_path = pathlib.Path(__file__).parent.parent
        with open(cls.fixture_path / 'fixtures/doc_fixture_0c027af0ee9c1901c57f6579d903aedee7f4.pkl',
                  'rb') as file_handle:
            doc_json = pickle.load(file_handle)
            cls.doc = Document.from_json(doc_json)

        with open(cls.fixture_path / 'fixtures/doc_fixture_a84d359922a69916a05de7c91204b79d02c36cda.pkl',
                  'rb') as file_handle:
            doc_json = pickle.load(file_handle)
            cls.doc_2 = Document.from_json(doc_json)
        assert cls.doc.pages
        assert cls.doc.tokens
        assert cls.doc_2.pages
        assert cls.doc_2.tokens
        cls.figure_table_predictions = FigureTablePredictions(cls.doc)
        cls.figure_table_predictions_2 = FigureTablePredictions(cls.doc_2)

    def test_merge_boxes(self):
        result = FigureTablePredictions._merge_boxes(self.doc.layoutparser_span_groups, defaultdict(list))
        assert isinstance(result, dict)
        assert list(result.keys()) == [0, 2, 3, 7]
        assert isinstance(result[0][0], Span)

    def test_get_figure_caption_distance(self):
        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == 900

        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == pytest.approx(0.15)

    def test_predict(self):
        result = self.figure_table_predictions.predict()
        assert isinstance(result, tuple)
        assert [entry.type for entry in result[1]] == ['Table', 'Table', 'Table', 'Table']
        assert [entry.type for entry in result[0]] == ['Figure', 'Figure', 'Figure', 'Figure']

    def test___cast_to_caption_vila_tokens(self):
        self.figure_table_predictions._cast_to_caption_vila_tokens()
        result = self.figure_table_predictions.vila_caption_dict
        assert isinstance(result, dict)
        assert set(result.keys()) == set([0, 2, 3, 7])
        assert set([len(item) for item in result.values()]) == set([34, 59, 12, 103])

    def test__filter_span_group(self):
        result_fig = [entry.text for entry in self.figure_table_predictions._filter_span_group(
            self.doc.vila_span_groups, 'fig', span_group_type='Caption')]

        assert result_fig == ['Figure 1: Motivation of our work. The content in the cur-\n'
                              'rent sliding window is a cluster of pixels of tree. We pro-\n'
                              'pose to incorporate geospatial knowledge to build a pooling\n'
                              'function which can propagate such a spatial',
                              'Figure 2: Given a feature map as an input, max pooling\n'
                              '(top right) and the proposed G -pooling (bottom right) cre-\n'
                              'ate different output downsampled feature map based on the\n'
                              'characteristics of spatial cluster. The feature map within\n'
                              'the sliding window (blue dot line) indicates a spatial clus-\n'
                              'ter. Max pooling takes the max value ignoring the spatial\n'
                              'cluster, while our G -pooling takes the interpolated value at\n'
                              'the center location. (White, gray and black represent three\n'
                              'values range from low to high.)',
                              'Figure 3: A FCN network architecture with G -pooling.',
                              'Figure 4: Qualitative results of ISPRS Potsdam. White: road, blue: building, '
                              'cyan: low vegetation, green: trees, yellow:\n'
                              'cars, red: clutter.']

        result_ = [entry.text for entry in self.figure_table_predictions._filter_span_group(
            self.doc.vila_span_groups, '', span_group_type='Caption')]

        assert result_ == ['Figure 1: Motivation of our work. The content in the cur-\n'
                           'rent sliding window is a cluster of pixels of tree. We pro-\n'
                           'pose to incorporate geospatial knowledge to build a pooling\n'
                           'function which can propagate such a spatial',
                           'Figure 2: Given a feature map as an input, max pooling\n'
                           '(top right) and the proposed G -pooling (bottom right) cre-\n'
                           'ate different output downsampled feature map based on the\n'
                           'characteristics of spatial cluster. The feature map within\n'
                           'the sliding window (blue dot line) indicates a spatial clus-\n'
                           'ter. Max pooling takes the max value ignoring the spatial\n'
                           'cluster, while our G -pooling takes the interpolated value at\n'
                           'the center location. (White, gray and black represent three\n'
                           'values range from low to high.)',
                           'Ord',
                           '∗',
                           'Figure 3: A FCN network architecture with G -pooling.',
                           'Table 1: Experimental results of FCN using VGG-16 as backbone. Stride conv, '
                           'P -pooling and ours G -pooling are used to\n'
                           'replaced the standard max/average pooling.',
                           'Table 2: Cross-location evaluation. We compare the generalization capability '
                           'of using G -pooling with domain adaptation\n'
                           'method AdaptSegNet which utilize the unlabeled data.',
                           'Table 3: The average percentage of detected spatial clusters\n'
                           'per feature map with different threshold.',
                           'SegNet',
                           'Table 4: Experimental results on comparing w/o and w/ pro-\n'
                           'posed G -pooling for the state-of-the-art segmentation net-\n'
                           'works. P → V indicates the model trained on Potsdam and\n'
                           'test on Vaihingen, and versa the verses.\n'
                           'Potsdam (P)\n'
                           'P →',
                           'Figure 4: Qualitative results of ISPRS Potsdam. White: road, blue: building, '
                           'cyan: low vegetation, green: trees, yellow:\n'
                           'cars, red: clutter.']

        result_tab = [entry.text for entry in self.figure_table_predictions._filter_span_group(
            self.doc.vila_span_groups, 'tab', span_group_type='Caption')]

        assert result_tab == ['Table 1: Experimental results of FCN using VGG-16 as backbone. Stride conv, '
                              'P -pooling and ours G -pooling are used to\n'
                              'replaced the standard max/average pooling.',
                              'Table 2: Cross-location evaluation. We compare the generalization capability '
                              'of using G -pooling with domain adaptation\n'
                              'method AdaptSegNet which utilize the unlabeled data.',
                              'Table 3: The average percentage of detected spatial clusters\n'
                              'per feature map with different threshold.',
                              'Table 4: Experimental results on comparing w/o and w/ pro-\n'
                              'posed G -pooling for the state-of-the-art segmentation net-\n'
                              'works. P → V indicates the model trained on Potsdam and\n'
                              'test on Vaihingen, and versa the verses.\n'
                              'Potsdam (P)\n'
                              'P →']

    def test__filter_span_group_doc_2_caption_2(self):
        result_negation = [[self.doc_2.symbols[entry.start: entry.end]] for span_group in
                           self.figure_table_predictions_2._filter_span_group(
                               self.doc_2.vila_span_groups, '', span_group_type='Caption', negation=False) for entry in
                           span_group]

        assert result_negation == [['Figure 2:\n'
                                    'Hypergradient computation. The entire\n'
                                    'computation can be performed eﬃciently using vector-\n'
                                    'Jacobian products, provided a cheap approximation to\n'
                                    'the inverse-Hessian-vector product is available.\n'
                                    'Algorithm 1 Gradient-based HO for λ ∗ , w ∗ ( λ ∗ )'],
                                   ['2 hypergradient ( L V , L T , λ (cid:48) , w'],
                                   [')'],
                                   ['Figure 3: Comparing approximate hypergradients for\n'
                                    'inverse Hessian approximations to true hypergradients.\n'
                                    'The Neumann'],
                                   ['often has greater cosine similarity\n'
                                    'than CG, but larger (cid:96) 2 distance for equal steps.'],
                                   ['Figure 4: Inverse Hessian approximations preprocessed\n'
                                    'by applying tanh for a 1-layer, fully-connected NN on\n'
                                    'the Boston housing dataset as in [50].'],
                                   ['Figure 5: Algorithm 1 can overﬁt a small validation\nset on'],
                                   ['.'],
                                   ['100 %'],
                                   ['for standard, large'],
                                   ['.']]

    def test__filter_span_group_doc_2_negation(self):
        result_negation = [[self.doc_2.symbols[entry.start: entry.end]] for span_group in
                           self.figure_table_predictions_2._filter_span_group(
                               self.doc_2.vila_span_groups, '', span_group_type='Caption', negation=True) for entry in
                           span_group]

        assert result_negation == [['Jonathan Lorraine, Paul Vicol, David Duvenaud'],
                                   ['∂ L ∗ V ∂ λ\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123) =\n'
                                    '∂ L V ∂ λ\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123)\n'
                                    '+\n'
                                    '∂ L V ∂ w\n'
                                    '(cid:122) (cid:125)(cid:124) (cid:123)\n'
                                    '∂ w ∗ ∂ λ\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123)\n'
                                    '=\n'
                                    '∂ L V ∂ λ\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123)\n'
                                    '+\n'
                                    '∂ L V ∂ w\n'
                                    '(cid:122) (cid:125)(cid:124) (cid:123)\n'
                                    '− (cid:2) ∂ 2 L T ∂ w ∂ w T (cid:3) − 1\n'
                                    '(cid:122) (cid:125)(cid:124) (cid:123)\n'
                                    '(cid:124)\n'
                                    '(cid:123)(cid:122)\n'
                                    '(cid:125) vector-inverse Hessian product\n'
                                    '∂ 2 L T ∂ w ∂ λ T\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123)\n'
                                    '=\n'
                                    '∂ L V ∂ λ\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123)\n'
                                    '+\n'
                                    '∂ L V ∂ w ×− (cid:2) ∂ 2 L T ∂ w ∂ w T (cid:3) − 1\n'
                                    '(cid:122) (cid:125)(cid:124) (cid:123)\n'
                                    '∂ 2 L T ∂ w ∂ λ T\n'
                                    '(cid:122)(cid:125)(cid:124)(cid:123)\n'
                                    '(cid:124)\n'
                                    '(cid:123)(cid:122)\n'
                                    '(cid:125) vector-Jacobian product'],
                                   ['1: Initialize hyperparameters λ (cid:48) and weights w (cid:48)\n'
                                    '2: while not converged do\n'
                                    '3:\n'
                                    'for k = 1 . . . N do\n'
                                    '4:\n'
                                    'w (cid:48) − = α · ∂ L T ∂ w | λ (cid:48) , w (cid:48)\n'
                                    '5:\n'
                                    'λ (cid:48) − = hypergradient ( L V , L T , λ (cid:48) , w (cid:48) )\n'
                                    '6: return λ (cid:48) , w (cid:48)\n'
                                    '(cid:46) λ ∗ , w ∗ ( λ ∗ ) from Eq.1'],
                                   ['Algorithm'],
                                   ['(cid:48)'],
                                   ['1: v 1 = ∂ L V ∂ w | λ (cid:48) , w (cid:48)\n'
                                    '2: v 2 = approxInverseHVP ( v 1 , ∂ L T ∂ w )\n'
                                    '3: v 3 = grad ( ∂ L T ∂ λ , w , grad_outputs = v 2 )\n'
                                    '4: return ∂ L V ∂ λ | λ (cid:48) , w (cid:48) − v 3\n'
                                    '(cid:46) Return to Alg. 1\n'
                                    'Algorithm 3 approxInverseHVP ( v , f ): Neumann ap-\n'
                                    'proximation of inverse-Hessian-vector product v [ ∂ f ∂ w ] − 1\n'
                                    '1: Initialize sum p = v\n'
                                    '2: for j = 1 . . . i do\n'
                                    '3:\n'
                                    'v − = α · grad ( f , w , grad_outputs = v )\n'
                                    '4:\n'
                                    'p − = v\n'
                                    '5: return p\n'
                                    '(cid:46) Return'],
                                   ['to Alg. 2'],
                                   ['.'],
                                   ['2.1\nProposed Algorithms'],
                                   ['We outline our method in Algs. 1, 2, and 3, where α\n'
                                    'denotes the learning rate. Alg. 3 is also shown in [22].\n'
                                    'We visualize the hypergradient computation in Fig. 2.'],
                                   ['3\nRelated Work'],
                                   ['Implicit Function Theorem. The IFT has been\n'
                                    'used for optimization in nested optimization prob-\n'
                                    'lems [27, 28], backpropagating through arbitrarily long\n'
                                    'RNNs [22], or even eﬃcient k -fold cross-validation [29].\n'
                                    'Early work applied the IFT to regularization by ex-\n'
                                    'plicitly computing the Hessian (or Gauss-Newton) in-\n'
                                    'verse [23, 2]. In [24], the identity matrix is used to ap-\n'
                                    'proximate the inverse Hessian in the IFT. HOAG [30]\n'
                                    'uses conjugate gradient (CG) to invert the Hessian\n'
                                    'approximately and provides convergence results given\n'
                                    'tolerances on the optimal parameter and inverse. In\n'
                                    'iMAML [9], a center to the weights is ﬁt to perform\n'
                                    'well on multiple tasks—contrasted with our use of vali-\n'
                                    'dation loss. In DEQ [31], implicit diﬀerentiation is used\n'
                                    'to add diﬀerentiable ﬁxed-point methods into NN ar-\n'
                                    'chitectures. We use a Neumann approximation for the\n'
                                    'inverse-Hessian, instead of CG [30, 9] or the identity.\n'
                                    'Approximate inversion algorithms. CG is diﬃ-\n'
                                    'cult to scale to modern, deep NNs. We use the Neu-\n'
                                    'mann inverse approximation, which was observed to\n'
                                    'be a stable alternative to CG in NNs [22, 7]. The\n'
                                    'stability is motivated by connections between the Neu-\n'
                                    'mann approximation and unrolled diﬀerentiation [7].\n'
                                    'Alternatively, we could use prior knowledge about the\n'
                                    'NN structure to aid in the inversion—e.g., by using\n'
                                    'KFAC [32]. It is possible to approximate the Hessian\n'
                                    'with the Gauss-Newton matrix or Fisher Information\n'
                                    'matrix [23]. Various works use an identity approxi-\n'
                                    'mation to the inverse, which is equivalent to 1 -step\n'
                                    'unrolled diﬀerentiation [24, 14, 33, 10, 8, 34, 35].\n'
                                    'Unrolled diﬀerentiation for HO. A key diﬃculty\n'
                                    'in nested optimization is approximating how the opti-\n'
                                    'mized inner parameters (i.e., NN weights) change with\n'
                                    'respect to the outer parameters (i.e., hyperparameters).\n'
                                    'We often optimize the inner parameters with gradient\n'
                                    'descent, so we can simply diﬀerentiate through this\n'
                                    'optimization. Diﬀerentiation through optimization has\n'
                                    'been applied to nested optimization problems by [3],\n'
                                    'was scaled to HO for NNs by [4], and has been applied\n'
                                    'to various applications like learning optimizers [36]. [6]\n'
                                    'provides convergence results for this class of algorithms,\n'
                                    'while [5] discusses forward- and reverse-mode variants.\n'
                                    'As the number of gradient steps we backpropagate\n'
                                    'through increases, so does the memory and computa-\n'
                                    'tional cost. Often, gradient descent does not exactly\n'
                                    'minimize our objective after a ﬁnite number of steps—\n'
                                    'it only approaches a local minimum. Thus, to see how\n'
                                    'the hyperparameters aﬀect the local minima, we may\n'
                                    'have to unroll the optimization infeasibly far. Unrolling\n'
                                    'a small number of steps can be crucial for performance\n'
                                    'but may induce bias [37]. [7] discusses connections\n'
                                    'between unrolling and the IFT, and proposes to un-\n'
                                    'roll only the last L -steps. DrMAD [38] proposes an\n'
                                    'interpolation scheme to save memory.\n'
                                    'We compare hypergradient approximations in Table 1,\n'
                                    'and memory costs of gradient-based HO methods in\n'
                                    'Table 2. We survey gradient-free HO in Appendix B.'],
                                   ['Jonathan Lorraine, Paul Vicol, David Duvenaud'],
                                   ['[46] James Bergstra and Yoshua Bengio. Random\n'
                                    'search for hyper-parameter optimization. Journal\n'
                                    'of Machine Learning Research , 13:281–305, 2012.\n'
                                    '[47] Manoj Kumar, George E Dahl, Vijay Vasudevan,\n'
                                    'and Mohammad Norouzi. Parallel architecture and\n'
                                    'hyperparameter search via successive halving and\n'
                                    'classiﬁcation. arXiv preprint arXiv:1805.10255 ,\n'
                                    '2018.\n'
                                    '[48] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin\n'
                                    'Rostamizadeh, and Ameet Talwalkar. Hyperband:\n'
                                    'a novel bandit-based approach to hyperparameter\n'
                                    'optimization. The Journal of Machine Learning\n'
                                    'Research , 18(1):6765–6816, 2017.\n'
                                    '[49] David Harrison Jr and Daniel L Rubinfeld. He-\n'
                                    'donic housing prices and the demand for clean\n'
                                    'air. Journal of Environmental Economics and\n'
                                    'Management , 5(1):81–102, 1978.\n'
                                    '[50] Guodong Zhang, Shengyang Sun, David Duve-\n'
                                    'naud, and Roger Grosse. Noisy natural gradient\n'
                                    'as variational inference. In International Con-\n'
                                    'ference on Machine Learning , pages 5847–5856,\n'
                                    '2018.\n'
                                    '[51] Alex Krizhevsky, Ilya Sutskever, and Geoﬀrey E\n'
                                    'Hinton. ImageNet classiﬁcation with deep convo-\n'
                                    'lutional neural networks. In Advances in Neural\n'
                                    'Information Processing Systems , pages 1097–1105,\n'
                                    '2012.\n'
                                    '[52] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and\n'
                                    'Jian Sun. Deep residual learning for image recog-\n'
                                    'nition. In Conference on Computer Vision and\n'
                                    'Pattern Recognition , pages 770–778, 2016.\n'
                                    '[53] Alex Krizhevsky. Learning multiple layers of fea-\n'
                                    'tures from tiny images. Technical report, 2009.\n'
                                    '[54] Olaf Ronneberger, Philipp Fischer, and Thomas\n'
                                    'Brox. U-Net: Convolutional networks for biomed-\n'
                                    'ical image segmentation. In International Confer-\n'
                                    'ence on Medical image Computing and Computer-\n'
                                    'Assisted Intervention , pages 234–241, 2015.\n'
                                    '[55] Saypraseuth Mounsaveng, David Vazquez, Is-\n'
                                    'mail Ben Ayed, and Marco Pedersoli. Adversarial\n'
                                    'learning of general transformations for data aug-\n'
                                    'mentation. International Conference on Learning\n'
                                    'Representations , 2019.\n'
                                    '[56] Sepp Hochreiter and Jürgen Schmidhuber. Long\n'
                                    'short-term memory. Neural Computation , 9(8):\n'
                                    '1735–1780, 1997.\n'
                                    '[57] Mitchell P Marcus, Mary Ann Marcinkiewicz, and\n'
                                    'Beatrice Santorini. Building a large annotated\n'
                                    'corpus of English: The Penn Treebank. Computa-\n'
                                    'tional Linguistics , 19(2):313–330, 1993.\n'
                                    '[58] Yarin Gal and Zoubin Ghahramani. A theoreti-\n'
                                    'cally grounded application of dropout in recurrent\n'
                                    'neural networks. In Advances in Neural Informa-\n'
                                    'tion Processing Systems , pages 1027–1035, 2016.\n'
                                    '[59] Durk P Kingma, Tim Salimans, and Max Welling.\n'
                                    'Variational dropout and the local reparameteri-\n'
                                    'zation trick. In Advances in Neural Information\n'
                                    'Processing Systems , pages 2575–2583, 2015.\n'
                                    '[60] Li Wan, Matthew Zeiler, Sixin Zhang, Yann\n'
                                    'Le Cun, and Rob Fergus. Regularization of neu-\n'
                                    'ral networks using Dropconnect. In International\n'
                                    'Conference on Machine Learning , pages 1058–1066,\n'
                                    '2013.\n'
                                    '[61] Yarin Gal, Jiri Hron, and Alex Kendall. Con-\n'
                                    'crete dropout. In Advances in Neural Information\n'
                                    'Processing Systems , pages 3581–3590, 2017.\n'
                                    '[62] Ian Goodfellow, Yoshua Bengio, and Aaron\n'
                                    'Courville.\n'
                                    'Deep Learning .\n'
                                    'MIT Press, 2016.\n'
                                    'http://www.deeplearningbook.org .\n'
                                    '[63] Jakob Foerster, Richard Y Chen, Maruan Al-\n'
                                    'Shedivat, Shimon Whiteson, Pieter Abbeel, and\n'
                                    'Igor Mordatch. Learning with opponent-learning\n'
                                    'awareness. In International Conference on Au-\n'
                                    'tonomous Agents and MultiAgent Systems , pages\n'
                                    '122–130, 2018.\n'
                                    '[64] Alistair Letcher, Jakob Foerster, David Balduzzi,\n'
                                    'Tim Rocktäschel, and Shimon Whiteson. Stable\n'
                                    'opponent shaping in diﬀerentiable games. arXiv\n'
                                    'preprint arXiv:1811.08469 , 2018.\n'
                                    '[65] Andrew Brock, Theodore Lim, James Millar\n'
                                    'Ritchie, and Nicholas J Weston. SMASH: One-\n'
                                    'Shot Model Architecture Search through Hyper-\n'
                                    'Networks. In International Conference on Learn-\n'
                                    'ing Representations , 2018.\n'
                                    '[66] Adam Paszke, Sam Gross, Soumith Chintala, Gre-\n'
                                    'gory Chanan, Edward Yang, Zachary DeVito, Zem-\n'
                                    'ing Lin, Alban Desmaison, Luca Antiga, and Adam\n'
                                    'Lerer. Automatic diﬀerentiation in PyTorch. 2017.\n'
                                    '[67] Diederik P Kingma and Jimmy Ba. Adam: A\n'
                                    'method for stochastic optimization. International\n'
                                    'Conference on Learning Representations , 2014.\n'
                                    '[68] Geoﬀrey Hinton, Nitish Srivastava, and Kevin\n'
                                    'Swersky. Neural networks for machine learning.\n'
                                    'Lecture 6a. Overview of mini-batch gradient de-\n'
                                    'scent. 2012.\n'
                                    '[69] Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick\n'
                                    'Haﬀner, et al. Gradient-based learning applied to\n'
                                    'document recognition. Proceedings of the IEEE ,\n'
                                    '86(11):2278–2324, 1998.\n'
                                    '[70] Stephen Merity, Nitish Shirish Keskar, and\n'
                                    'Richard Socher.\n'
                                    'Regularizing and optimizing\n'
                                    'LSTM language models. International Confer-\n'
                                    'ence on Learning Representations , 2018.'],
                                   ['Jonathan Lorraine, Paul Vicol, David Duvenaud'],
                                   ['0.0\n'
                                    '0.2\n'
                                    '0.4\n'
                                    '0.6\n'
                                    '0.8\n'
                                    '1.0\n'
                                    '400\n'
                                    '600 10 3\n'
                                    '10 1\n'
                                    '10 1\n'
                                    '10 3\n'
                                    '20 CG Steps 20 Neumann\n'
                                    '5 Neumann 1 Neumann\n'
                                    '0 Neumann'],
                                   ['C o s i\nn e\nS i m\nil a r i t\ny\n(cid:96)'],
                                   ['2'],
                                   ['D i s t\na n ce\nOptimization Iter.'],
                                   ['0\n20\n40\nCGNeumann\n# of Inversion Steps'],
                                   ['scheme'],
                                   ['1.0\n'
                                    '0.5\n'
                                    '0.0\n'
                                    '0.5\n'
                                    '1.0\n'
                                    '1 Neumann\n'
                                    '1.0\n'
                                    '0.5\n'
                                    '0.0\n'
                                    '0.5\n'
                                    '1.0\n'
                                    '5 Neumann\n'
                                    '1.0\n'
                                    '0.5\n'
                                    '0.0\n'
                                    '0.5\n'
                                    '1.0\n'
                                    'True Inverse\n'
                                    '1.0\n'
                                    '0.5\n'
                                    '0.0\n'
                                    '0.5\n'
                                    '1.0'],
                                   ['5.2\nOverﬁtting a Small Validation Set'],
                                   ['In Fig. 5, we check the capacity of our HO algorithm\n'
                                    'to overﬁt the validation dataset. We use the same re-\n'
                                    'stricted dataset as in [5, 6] of 50 training and validation\n'
                                    'examples, which allows us to assess HO performance\n'
                                    'easily. We tune a separate weight decay hyperparam-\n'
                                    'eter for each NN parameter as in [33, 4]. We show\n'
                                    'the performance with a linear classiﬁer, AlexNet [51],\n'
                                    'and ResNet 44 [52]. For AlexNet, this yields more than\n'
                                    '50 000 000 hyperparameters, so we can perfectly classify\n'
                                    'our validation data by optimizing the hyperparameters.\n'
                                    'Algorithm 1 achieves 100 % accuracy on the training\n'
                                    'and validation sets with signiﬁcantly lower accuracy\n'
                                    'on the test set (Appendix E, Fig. 10), showing that we\n'
                                    'have a powerful HO algorithm. The same optimizer is\n'
                                    'used for weights and hyperparameters in all cases.'],
                                   ['5.3\nDataset Distillation'],
                                   ['Dataset distillation [4, 13] aims to learn a small, syn-\n'
                                    'thetic training dataset from scratch, that condenses the\n'
                                    'knowledge contained in the original full-sized training\n'
                                    'set. The goal is that a model trained on the synthetic\n'
                                    '10 0\n'
                                    '10 1\n'
                                    '10 2\n'
                                    '10 3\n'
                                    '10 4 0.0\n'
                                    '0.2\n'
                                    '0.4\n'
                                    '0.6\n'
                                    '0.8\n'
                                    '1.0\n'
                                    'Linear\n'
                                    'AlexNet\n'
                                    'ResNet44\n'
                                    'V a li\n'
                                    'd a t i o\n'
                                    'n E\n'
                                    'rr o r\n'
                                    'Iteration'],
                                   ['CIFAR- 10'],
                                   ['It optimizes for loss and achieves'],
                                   ['validation accuracy'],
                                   ['models'],
                                   ['data generalizes to the original validation and test sets.\n'
                                    'Distillation is an interesting benchmark for HO as it\n'
                                    'allows us to introduce tens of thousands of hyperparam-\n'
                                    'eters, and visually inspect what is learned: here, every\n'
                                    'pixel value in each synthetic training example is a hyper-\n'
                                    'parameter. We distill MNIST and CIFAR- 10 / 100 [53],\n'
                                    'yielding 28 × 28 × 10 = 7840 , 32 × 32 × 3 × 10 = 30 720 , and\n'
                                    '32 × 32 × 3 × 100 = 300 720 hyperparameters, respectively.\n'
                                    'For these experiments, all labeled data are in our vali-\n'
                                    'dation set, while our distilled data are in the training\n'
                                    'set. We visualize the distilled images for each class in\n'
                                    'Fig. 6, recovering recognizable digits for MNIST and\n'
                                    'reasonable color averages for CIFAR- 10 / 100 .'],
                                   ['5.4\nLearned Data Augmentation'],
                                   ['Data augmentation is a simple way to introduce invari-\n'
                                    'ances to a model—such as scale or contrast invariance—\n'
                                    'that improve generalization [17, 18]. Taking advantage\n'
                                    'of the ability to optimize many hyperparameters, we\n'
                                    'learn data augmentation from scratch (Fig. 7).\n'
                                    'Speciﬁcally, we learn a data augmentation network\n'
                                    '˜x = f λ ( x , (cid:15) ) that takes a training example x and noise\n'
                                    '(cid:15) ∼ N (0 , I ) , and outputs an augmented example ˜x .\n'
                                    'The noise (cid:15) allows us to learn stochastic augmentations.\n'
                                    'We parameterize f as a U-net [54] with a residual con-\n'
                                    'nection from the input to the output, to make it easy\n'
                                    'to learn the identity mapping. The parameters of the\n'
                                    'U-net, λ , are hyperparameters tuned for the validation\n'
                                    'loss—thus, we have 6659 hyperparameters. We trained\n'
                                    'a ResNet 18 [52] on CIFAR-10 with augmented exam-\n'
                                    'ples produced by the U-net (that is simultaneously\n'
                                    'trained on the validation set).\n'
                                    'Results for the identity and Neumann inverse approxi-\n'
                                    'mations are shown in Table 3. We omit CG because it\n'
                                    'performed no better than the identity. We found that\n'
                                    'using the data augmentation network improves vali-\n'
                                    'dation and test accuracy by 2-3%, and yields smaller\n'
                                    'variance between multiple random restarts. In [55], a\n'
                                    'diﬀerent augmentation network architecture is learned\n'
                                    'with adversarial training.']]

    def test__filter_span_group_2(self):
        span_groups_to_filter_json = [{'metadata': {'type': 'Title'}, 'spans': [{'end': 86, 'start': 0}]},
                                      {'metadata': {'type': 'Author'}, 'spans': [{'end': 248, 'start': 87}]},
                                      {'metadata': {'type': 'Abstract'}, 'spans': [{'end': 1546, 'start': 249}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 1562, 'start': 1547}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 2033, 'start': 1563}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 3408, 'start': 2256}]},
                                      {'metadata': {'type': 'Footer'}, 'spans': [{'end': 3464, 'start': 3409}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 5496, 'start': 3465}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 5512, 'start': 5497}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 8996, 'start': 5513}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 9007, 'start': 8997}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9174, 'start': 9008}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9739, 'start': 9680}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9745, 'start': 9744}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9944, 'start': 9748}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 9946, 'start': 9945}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9947, 'start': 9946}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 9987, 'start': 9947}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 12432, 'start': 9988}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 12434, 'start': 12432}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13512, 'start': 12435}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13515, 'start': 13513}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13520, 'start': 13516}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13624, 'start': 13521}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13630, 'start': 13625}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13631, 'start': 13630}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13636, 'start': 13631}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13640, 'start': 13637}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13660, 'start': 13641}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13663, 'start': 13661}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13676, 'start': 13664}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13821, 'start': 13677}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13905, 'start': 13876}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13991, 'start': 13906}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 14555, 'start': 13991}]},
                                      {'metadata': {'type': 'Equation'}, 'spans': [{'end': 14564, 'start': 14556}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 15640, 'start': 14565}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 15666, 'start': 15641}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 16665, 'start': 15667}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 16699, 'start': 16666}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 17391, 'start': 16700}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 17401, 'start': 17392}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 18197, 'start': 17402}]},
                                      {'metadata': {'type': 'Table'}, 'spans': [{'end': 19521, 'start': 18360}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 20115, 'start': 19521}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 20139, 'start': 20116}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 21127, 'start': 20140}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 21129, 'start': 21128}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 21130, 'start': 21129}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 21155, 'start': 21130}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 22041, 'start': 21156}]},
                                      {'metadata': {'type': 'Table'}, 'spans': [{'end': 23399, 'start': 22215}]},
                                      {'metadata': {'type': 'Table'}, 'spans': [{'end': 23575, 'start': 23503}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 23606, 'start': 23576}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 24427, 'start': 23607}]},
                                      {'metadata': {'type': 'Table'}, 'spans': [{'end': 24464, 'start': 24428}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 26240, 'start': 24465}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 26994, 'start': 26241}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 27037, 'start': 26995}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 29360, 'start': 27038}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 29368, 'start': 29367}]},
                                      {'metadata': {'type': 'Table'}, 'spans': [{'end': 30049, 'start': 29601}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 30063, 'start': 30050}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 31223, 'start': 30064}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 33685, 'start': 31365}]},
                                      {'metadata': {'type': 'Section'}, 'spans': [{'end': 33699, 'start': 33686}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 33871, 'start': 33700}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 34495, 'start': 33872}]},
                                      {'metadata': {'type': 'Bibliography'}, 'spans': [{'end': 39499, 'start': 34496}]},
                                      {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 39501, 'start': 39500}]},
                                      {'metadata': {'type': 'Bibliography'}, 'spans': [{'end': 42268, 'start': 39502}]}]

        span_groups_to_filter = [mmda_ann.SpanGroup.from_json(entry) for entry in span_groups_to_filter_json]
        result = FigureTablePredictions._filter_span_group(
            span_groups_to_filter, caption_content='', span_group_type='Paragraph')
        assert [entry.to_json() for entry in result] == [
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 2033, 'start': 1563}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 3408, 'start': 2256}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 5496, 'start': 3465}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 8996, 'start': 5513}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9174, 'start': 9008}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9739, 'start': 9680}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9745, 'start': 9744}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9944, 'start': 9748}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 9947, 'start': 9946}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 12432, 'start': 9988}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13512, 'start': 12435}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13520, 'start': 13516}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13630, 'start': 13625}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13636, 'start': 13631}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13660, 'start': 13641}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13676, 'start': 13664}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 13905, 'start': 13876}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 14555, 'start': 13991}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 15640, 'start': 14565}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 16665, 'start': 15667}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 17391, 'start': 16700}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 18197, 'start': 17402}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 20115, 'start': 19521}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 21127, 'start': 20140}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 21130, 'start': 21129}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 22041, 'start': 21156}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 24427, 'start': 23607}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 26240, 'start': 24465}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 26994, 'start': 26241}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 29360, 'start': 27038}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 29368, 'start': 29367}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 31223, 'start': 30064}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 33685, 'start': 31365}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 33871, 'start': 33700}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 34495, 'start': 33872}]},
            {'metadata': {'type': 'Paragraph'}, 'spans': [{'end': 39501, 'start': 39500}]}]

        result = FigureTablePredictions._filter_span_group(
            span_groups_to_filter, caption_content='', span_group_type='Paragraph', negation=True)
        assert [entry.to_json() for entry in result] == [
            {'metadata': {'type': 'Title'}, 'spans': [{'end': 86, 'start': 0}]},
            {'metadata': {'type': 'Author'}, 'spans': [{'end': 248, 'start': 87}]},
            {'metadata': {'type': 'Abstract'}, 'spans': [{'end': 1546, 'start': 249}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 1562, 'start': 1547}]},
            {'metadata': {'type': 'Footer'}, 'spans': [{'end': 3464, 'start': 3409}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 5512, 'start': 5497}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 9007, 'start': 8997}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 9946, 'start': 9945}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 9987, 'start': 9947}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 12434, 'start': 12432}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13515, 'start': 13513}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13624, 'start': 13521}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13631, 'start': 13630}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13640, 'start': 13637}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13663, 'start': 13661}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13821, 'start': 13677}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 13991, 'start': 13906}]},
            {'metadata': {'type': 'Equation'}, 'spans': [{'end': 14564, 'start': 14556}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 15666, 'start': 15641}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 16699, 'start': 16666}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 17401, 'start': 17392}]},
            {'metadata': {'type': 'Table'}, 'spans': [{'end': 19521, 'start': 18360}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 20139, 'start': 20116}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 21129, 'start': 21128}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 21155, 'start': 21130}]},
            {'metadata': {'type': 'Table'}, 'spans': [{'end': 23399, 'start': 22215}]},
            {'metadata': {'type': 'Table'}, 'spans': [{'end': 23575, 'start': 23503}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 23606, 'start': 23576}]},
            {'metadata': {'type': 'Table'}, 'spans': [{'end': 24464, 'start': 24428}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 27037, 'start': 26995}]},
            {'metadata': {'type': 'Table'}, 'spans': [{'end': 30049, 'start': 29601}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 30063, 'start': 30050}]},
            {'metadata': {'type': 'Section'}, 'spans': [{'end': 33699, 'start': 33686}]},
            {'metadata': {'type': 'Bibliography'}, 'spans': [{'end': 39499, 'start': 34496}]},
            {'metadata': {'type': 'Bibliography'}, 'spans': [{'end': 42268, 'start': 39502}]}]

        result = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups, caption_content='fig', span_group_type='Caption', negation=False)
        assert [[entry.text] for entry in result] == [
            ['Figure 1: Motivation of our work. The content in the cur-\n'
             'rent sliding window is a cluster of pixels of tree. We pro-\n'
             'pose to incorporate geospatial knowledge to build a pooling\n'
             'function which can propagate such a spatial'],
            ['Figure 2: Given a feature map as an input, max pooling\n'
             '(top right) and the proposed G -pooling (bottom right) cre-\n'
             'ate different output downsampled feature map based on the\n'
             'characteristics of spatial cluster. The feature map within\n'
             'the sliding window (blue dot line) indicates a spatial clus-\n'
             'ter. Max pooling takes the max value ignoring the spatial\n'
             'cluster, while our G -pooling takes the interpolated value at\n'
             'the center location. (White, gray and black represent three\n'
             'values range from low to high.)'],
            ['Figure 3: A FCN network architecture with G -pooling.'],
            [
                'Figure 4: Qualitative results of ISPRS Potsdam. White: road, blue: '
                'building, cyan: low vegetation, green: trees, yellow:\n'
                'cars, red: clutter.']]

        result = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups, caption_content='tab', span_group_type='Caption', negation=False)
        assert [[entry.text] for entry in result] == [
            ['Table 1: Experimental results of FCN using VGG-16 as backbone. Stride conv, '
             'P -pooling and ours G -pooling are used to\n'
             'replaced the standard max/average pooling.'],
            ['Table 2: Cross-location evaluation. We compare the generalization '
             'capability of using G -pooling with domain adaptation\n'
             'method AdaptSegNet which utilize the unlabeled data.'],
            ['Table 3: The average percentage of detected spatial clusters\n'
             'per feature map with different threshold.'],
            ['Table 4: Experimental results on comparing w/o and w/ pro-\n'
             'posed G -pooling for the state-of-the-art segmentation net-\n'
             'works. P → V indicates the model trained on Potsdam and\n'
             'test on Vaihingen, and versa the verses.\n'
             'Potsdam (P)\n'
             'P →']]

    def test_merge_spans(self):
        result = []
        for _, value in self.figure_table_predictions.merge_vila_token_spans(caption_content='fig').items():
            fig_text = []
            for entry in value:
                fig_text.append(self.figure_table_predictions.doc.symbols[entry.start: entry.end])

            result.append(fig_text)
        assert result == [['Figure 1: Motivation of our work. The content in the cur-\n'
                           'rent sliding window is a cluster of pixels of tree. We pro-\n'
                           'pose to incorporate geospatial knowledge to build a pooling\n'
                           'function which can propagate such a spatial cluster during\n'
                           'training, while the standard pooling is not able to achieve it.'],
                          ['Figure 2: Given a feature map as an input, max pooling\n'
                           '(top right) and the proposed G -pooling (bottom right) cre-\n'
                           'ate different output downsampled feature map based on the\n'
                           'characteristics of spatial cluster. The feature map within\n'
                           'the sliding window (blue dot line) indicates a spatial clus-\n'
                           'ter. Max pooling takes the max value ignoring the spatial\n'
                           'cluster, while our G -pooling takes the interpolated value at\n'
                           'the center location. (White, gray and black represent three\n'
                           'values range from low to high.)'],
                          ['Figure 3: A FCN network architecture with G -pooling.'],
                          ['Figure 4: Qualitative results of ISPRS Potsdam. White: road, blue: '
                           'building, cyan: low vegetation, green: trees, yellow:\n'
                           'cars, red: clutter.']]

        result = []

        for _, value in self.figure_table_predictions.merge_vila_token_spans(caption_content='tab').items():
            page_entities_text = []
            for entry in value:
                page_entities_text.append(self.figure_table_predictions.doc.symbols[entry.start: entry.end])

            result.append(page_entities_text)
        assert result == [['Table 1: Experimental results of FCN using VGG-16 as backbone. Stride conv, '
                           'P -pooling and ours G -pooling are used to\n'
                           'replaced the standard max/average pooling.'],
                          ['Table 2: Cross-location evaluation. We compare the generalization '
                           'capability of using G -pooling with domain adaptation\n'
                           'method AdaptSegNet which utilize the unlabeled data.',
                           'Table 3: The average percentage of detected spatial clusters\n'
                           'per feature map with different threshold.'],
                          ['Table 4: Experimental results on comparing w/o and w/ pro-\n'
                           'posed G -pooling for the state-of-the-art segmentation net-\n'
                           'works. P → V indicates the model trained on Potsdam and\n'
                           'test on Vaihingen, and versa the verses.',
                           'Potsdam (P)',
                           'P →']]

    def test__cast_to_caption_vila_tokens(self):
        # self.figure_table_predictions_2._cast_to_caption_vila_tokens(caption_content='tab')
        # assert self.figure_table_predictions_2.vila_caption_dict == defaultdict(list)

        self.figure_table_predictions_2._cast_to_caption_vila_tokens(caption_content='fig')
        result = [[self.figure_table_predictions_2.doc.symbols[entry.start: entry.end]]
                  for list_of_spans in self.figure_table_predictions_2.vila_caption_dict.values()
                  for entry in list_of_spans]
        assert result == []

    def test_merge_spans(self):
        result = [entry.to_json()
                  for list_of_spans in self.figure_table_predictions_2.merge_vila_token_spans(caption_content='fig').values()
                  for entry in list_of_spans]
        assert result == [{'box': {'height': 0.057863131313131255,
                                   'left': 0.12310620915032679,
                                   'page': 0,
                                   'top': 0.3078107883838384,
                                   'width': 0.38549067004039206},
                           'end': 1007,
                           'start': 805},
                          {'box': {'height': 0.014103064646464603,
                                   'left': 0.1228186274509804,
                                   'page': 0,
                                   'top': 0.390094844949495,
                                   'width': 0.35189126656862746},
                           'end': 1059,
                           'start': 1008},
                          {'box': {'height': 0.04276969696969718,
                                   'left': 0.12294444444444445,
                                   'page': 2,
                                   'top': 0.30825018232323226,
                                   'width': 0.38609548719588244},
                           'end': 10105,
                           'start': 9981},
                          {'box': {'height': 0.04276969696969696,
                                   'left': 0.12352941176470587,
                                   'page': 2,
                                   'top': 0.49324386919191915,
                                   'width': 0.3823604196639543},
                           'end': 10458,
                           'start': 10312},
                          {'box': {'height': 0.042768434343434414,
                                   'left': 0.5352941176470588,
                                   'page': 2,
                                   'top': 0.2575557378787879,
                                   'width': 0.38235892624790835},
                           'end': 11821,
                           'start': 11663}]

        result_text = [[self.figure_table_predictions_2.doc.symbols[entry['start']: entry['end']]] for entry in result]

        assert result_text == [['Figure 2:\n'
                                'Hypergradient computation. The entire\n'
                                'computation can be performed eﬃciently using vector-\n'
                                'Jacobian products, provided a cheap approximation to\n'
                                'the inverse-Hessian-vector product is available.'],
                               ['Algorithm 1 Gradient-based HO for λ ∗ , w ∗ ( λ ∗ )'],
                               ['Figure 3: Comparing approximate hypergradients for\n'
                                'inverse Hessian approximations to true hypergradients.\n'
                                'The Neumann scheme often has greate cosine similarity \n'
                                'than CG, but larger'],
                               ['Figure 4: Inverse Hessian approximations preprocessed\n'
                                'by applying tanh for a 1-layer, fully-connected NN on\n'
                                'the Boston housing dataset as in [50].'],
                               ['Figure 5: Algorithm 1 can overﬁt a small validation\n'
                                'set on CIFAR- 10 . It optimizes for loss and achieves\n'
                                '100 % validation accuracy for standard, large models']]
