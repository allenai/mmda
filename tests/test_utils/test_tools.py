"""


@kylel

"""

import json
import pathlib
import unittest

from mmda.types.annotation import BoxGroup, SpanGroup
from mmda.types.span import Span
from mmda.types.box import Box
from mmda.types.document import Document

from mmda.utils.tools import MergeSpans
from mmda.utils.tools import box_groups_to_span_groups

fixture_path = pathlib.Path(__file__).parent.parent / "fixtures" / "utils"


class TestMergeNeighborSpans(unittest.TestCase):

    def test_merge_multiple_neighbor_spans(self):
        spans = [Span(start=0, end=10), Span(start=11, end=20), Span(start=21, end=30)]
        merge_spans = MergeSpans(list_of_spans=spans, index_distance=1)
        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert len(out) == 1
        assert isinstance(out[0], Span)
        assert out[0].start == 0
        assert out[0].end == 30

    def test_different_index_distances(self):
        spans = [Span(start=0, end=10), Span(start=15, end=20)]
        merge_spans = MergeSpans(list_of_spans=spans, index_distance=1)
        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert out == spans  # no merge happened

        merge_spans = MergeSpans(list_of_spans=spans, index_distance=2)
        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert out == spans  # no merge happened

        merge_spans = MergeSpans(list_of_spans=spans, index_distance=4)
        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert out == spans  # no merge happened

        merge_spans = MergeSpans(list_of_spans=spans, index_distance=5)
        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert len(out) == 1
        assert isinstance(out[0], Span)
        assert out[0].start == 0
        assert out[0].end == 20

    def test_zero_index_distance(self):
        spans = [Span(start=0, end=10), Span(start=10, end=20)]
        out = MergeSpans(list_of_spans=spans, index_distance=0).merge_neighbor_spans_by_symbol_distance()
        assert len(out) == 1
        assert isinstance(out[0], Span)
        assert out[0].start == 0
        assert out[0].end == 20

    def test_handling_of_boxes(self):
        spans = [
            Span(start=0, end=10, box=Box(l=0, t=0, w=1, h=1, page=0)),
            Span(start=11, end=20, box=Box(l=1, t=1, w=2, h=2, page=0)),
            Span(start=21, end=150, box=Box(l=2, t=2, w=3, h=3, page=1))
        ]
        merge_spans = MergeSpans(list_of_spans=spans, index_distance=1)
        merge_spans.merge_neighbor_spans_by_symbol_distance()

        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert len(out) == 2
        assert isinstance(out[0], Span)
        assert isinstance(out[1], Span)
        assert out[0].start == 0
        assert out[0].end == 20
        assert out[1].start == 21
        assert out[1].end == 150
        assert out[0].box == Box(l=0, t=0, w=3, h=3, page=0)
        # unmerged spans from separate pages keep their original box
        assert out[1].box == spans[-1].box

        spans = [
            Span(start=0, end=10, box=Box(l=0, t=0, w=1, h=1, page=1)),
            Span(start=11, end=20, box=Box(l=1, t=1, w=2, h=2, page=1)),
            Span(start=100, end=150, box=Box(l=2, t=2, w=3, h=3, page=1))
        ]
        merge_spans = MergeSpans(list_of_spans=spans, index_distance=1)

        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert len(out) == 2
        assert isinstance(out[0], Span)
        assert isinstance(out[1], Span)
        assert out[0].start == 0
        assert out[0].end == 20
        assert out[1].start == 100
        assert out[1].end == 150
        assert out[0].box == Box(l=0, t=0, w=3, h=3, page=1)
        # unmerged spans that were too far apart in symbol distance keep their original box
        assert out[1].box == spans[-1].box

        spans = [
            Span(start=0, end=10, box=Box(l=0, t=0, w=1, h=1, page=0)),
            Span(start=11, end=20),
            Span(start=21, end=150),
            Span(start=155, end=200)
        ]
        merge_spans = MergeSpans(list_of_spans=spans, index_distance=1)
        merge_spans.merge_neighbor_spans_by_symbol_distance()

        out = merge_spans.merge_neighbor_spans_by_symbol_distance()
        assert len(out) == 3
        assert isinstance(out[0], Span)
        assert isinstance(out[1], Span)
        assert out[0].start == 0
        assert out[0].end == 10
        assert out[1].start == 11
        assert out[1].end == 150
        # spans without boxes are able to group together
        assert out[1].box is None
        # or not
        assert out[2].start == 155
        assert out[2].end == 200
        assert out[1].box is None


list_of_spans_to_merge = [
        Span(start=3944, end=3948,
             box=Box(l=0.19238134915568578, t=0.22752901673615306, w=0.06941334053447479, h=0.029442207414270286,
                     page=4)),
        Span(start=3949, end=3951,
             box=Box(l=0.27220460878651254, t=0.22752901673615306, w=0.03468585042904468, h=0.029442207414270286,
                     page=4)),
        Span(start=4060, end=4063,
             box=Box(l=0.4204075769894973, t=0.34144142726484455, w=0.023417310961637895, h=0.014200429984914883,
                     page=4)),
        Span(start=4072, end=4075,
             box=Box(l=0.5182742633669088, t=0.34144142726484455, w=0.029000512031393755, h=0.014200429984914883,
                     page=4)),
        Span(start=4076, end=4083,
             box=Box(l=0.5522956396696659, t=0.34144142726484455, w=0.06440764687304719, h=0.014200429984914883,
                     page=4)),
        Span(start=4119, end=4128,
             box=Box(l=0.2686971421659869, t=0.36273518298114954, w=0.08479235581478171, h=0.014200429984914883,
                     page=4)),
        Span(start=4134, end=4144,
             box=Box(l=0.40387889180816966, t=0.36273518298114954, w=0.08368776567508182, h=0.014200429984914883,
                     page=4)),
        Span(start=4145, end=4148,
             box=Box(l=0.4943548659781345, t=0.36273518298114954, w=0.042396177907390975, h=0.014200429984914883,
                     page=4)),
        Span(start=4149, end=4162,
             box=Box(l=0.5435392523804085, t=0.36273518298114954, w=0.11491754144296094, h=0.014200429984914883,
                     page=4)),
        Span(start=4166, end=4177,
             box=Box(l=0.6876581404256177, t=0.36273518298114954, w=0.09146006356715199, h=0.014200429984914883,
                     page=4)),
        Span(start=4419, end=4427,
             box=Box(l=0.2686971421659869, t=0.4479113936500019, w=0.06846450520430858, h=0.014200429984914883,
                     page=4)),
        Span(start=4497, end=4505,
             box=Box(l=0.2686971421659869, t=0.46920514936630686, w=0.06846450520430858, h=0.014200429984914883,
                     page=4)),
        Span(start=4517, end=4520,
             box=Box(l=0.42195400318507725, t=0.46920514936630686, w=0.029000512031393755, h=0.014200429984914883,
                     page=4)),
        Span(start=4574, end=4581,
             box=Box(l=0.2686971421659869, t=0.49049890508261185, w=0.07810456460532592, h=0.014200429984914883,
                     page=4)),
        Span(start=4582, end=4587,
             box=Box(l=0.35061756361754887, t=0.49049890508261185, w=0.03904224057412029, h=0.014200429984914883,
                     page=4)),
        Span(start=4588, end=4591,
             box=Box(l=0.39347566103790516, t=0.49049890508261185, w=0.023417310961637943, h=0.014200429984914883,
                     page=4)),
        Span(start=4592, end=4601,
             box=Box(l=0.4207088288457791, t=0.49049890508261185, w=0.08254300862121101, h=0.014200429984914883,
                     page=4)),
        Span(start=4602, end=4613,
             box=Box(l=0.5070676943132262, t=0.49049890508261185, w=0.09481400090042272, h=0.014200429984914883,
                     page=4)),]

list_of_spans_to_merge_2 = [Span(start=30113, end=30119,
                                 box=Box(l=0.12095229775767885, t=0.3578497466414853, w=0.05243790645011725,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30120, end=30124,
                                 box=Box(l=0.17929474059091924, t=0.3578497466414853, w=0.030687522426571887,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30125, end=30129,
                                 box=Box(l=0.21799556239458678, t=0.3578497466414853, w=0.04350076804709073,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30130, end=30135,
                                 box=Box(l=0.26740086682480063, t=0.3578497466414853, w=0.050208642713631964,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30136, end=30141,
                                 box=Box(l=0.32351404592155575, t=0.3578497466414853, w=0.0446254416438761,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30142, end=30151,
                                 box=Box(l=0.37404402394855496, t=0.3578497466414853, w=0.0769598075514552,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30152, end=30155,
                                 box=Box(l=0.4569284513402187, t=0.3578497466414853, w=0.029000512031393852,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30156, end=30165,
                                 box=Box(l=0.4918334997547357, t=0.3578497466414853, w=0.0792091547450259,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30166, end=30175,
                                 box=Box(l=0.5769471908828846, t=0.3578497466414853, w=0.07175819216632291,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30176, end=30179,
                                 box=Box(l=0.6576023545380633, t=0.3578497466414853, w=0.03122977576787907,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30180, end=30184,
                                 box=Box(l=0.6947366666890655, t=0.3578497466414853, w=0.03904224057412024,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30185, end=30190,
                                 box=Box(l=0.7396834436463088, t=0.3578497466414853, w=0.05020864271363187,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30191, end=30193,
                                 box=Box(l=0.7957966227430638, t=0.3578497466414853, w=0.015624929612482252,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30194, end=30197,
                                 box=Box(l=0.12095229775767885, t=0.37500875791374183, w=0.024541984558423317,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30198, end=30207,
                                 box=Box(l=0.1518205712980198, t=0.37500875791374183, w=0.07695980755145514,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30208, end=30210,
                                 box=Box(l=0.2351066678313926, t=0.37500875791374183, w=0.013395665875996984,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30211, end=30214,
                                 box=Box(l=0.2548286226893072, t=0.37500875791374183, w=0.02231272082193805,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30215, end=30217,
                                 box=Box(l=0.283467632493163, t=0.37500875791374183, w=0.015624929612482252,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30218, end=30221,
                                 box=Box(l=0.3054188510875629, t=0.37500875791374183, w=0.024541984558423317,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30222, end=30229,
                                 box=Box(l=0.33628712462790383, t=0.37500875791374183, w=0.055570925755447906,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30230, end=30235,
                                 box=Box(l=0.3981843393652693, t=0.37500875791374183, w=0.04183384110899822,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30236, end=30240,
                                 box=Box(l=0.44668588822663785, t=0.37500875791374183, w=0.03570838669793504,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30241, end=30244,
                                 box=Box(l=0.4887205639064905, t=0.37500875791374183, w=0.020083457085452783,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30245, end=30255,
                                 box=Box(l=0.5151303099738609, t=0.37500875791374183, w=0.08810612623388145,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30256, end=30259,
                                 box=Box(l=0.6095627251896601, t=0.37500875791374183, w=0.022312720821938,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30260, end=30262,
                                 box=Box(l=0.6382017349935157, t=0.37500875791374183, w=0.015624929612482252,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30263, end=30268,
                                 box=Box(l=0.6601529535879158, t=0.37500875791374183, w=0.03958449391542752,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30269, end=30273,
                                 box=Box(l=0.7098795933314969, t=0.37500875791374183, w=0.035708386697935225,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30274, end=30276,
                                 box=Box(l=0.7519142690113497, t=0.37500875791374183, w=0.013395665875997033,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30277, end=30278,
                                 box=Box(l=0.7716362238692644, t=0.37500875791374183, w=0.008917054945941066,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30279, end=30281,
                                 box=Box(l=0.7868795677971232, t=0.37500875791374183, w=0.02454198455842322,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30282, end=30291,
                                 box=Box(l=0.12095229775767885, t=0.3921677691859983, w=0.08031374488472577,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30292, end=30296,
                                 box=Box(l=0.2062869069137678, t=0.3921677691859983, w=0.03904224057412024,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30297, end=30302,
                                 box=Box(l=0.25035001175925126, t=0.3921677691859983, w=0.050208642713631964,
                                         h=0.014200429984914883,
                                         page=19)),
                            Span(start=30303, end=30311,
                                 box=Box(l=0.30557951874424644, t=0.3921677691859983, w=0.08143841848151108,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30312, end=30314,
                                 box=Box(l=0.3920388014971207, t=0.3921677691859983, w=0.016729519752182193,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30315, end=30321,
                                 box=Box(l=0.4137891855206661, t=0.3921677691859983, w=0.0535625800469026,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30322, end=30328,
                                 box=Box(l=0.47237262983893197, t=0.3921677691859983, w=0.05354249658981717,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30329, end=30333,
                                 box=Box(l=0.5309359907001122, t=0.3921677691859983, w=0.03681297683763493,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30334, end=30336,
                                 box=Box(l=0.5727698318091105, t=0.3921677691859983, w=0.01672951975218224,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30337, end=30344,
                                 box=Box(l=0.5945202158326559, t=0.3921677691859983, w=0.060230287799273016,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30345, end=30348,
                                 box=Box(l=0.6597713679032922, t=0.3921677691859983, w=0.029000512031393946,
                                         h=0.014200429984914883, page=19)),
                            Span(start=30349, end=30359,
                                 box=Box(l=0.6937927442060494, t=0.3921677691859983, w=0.07834556609035141,
                                         h=0.014200429984914883, page=19))]


def test_merge_spans():
    assert len(list_of_spans_to_merge) == (len(MergeSpans(list_of_spans_to_merge, 0, 0)
                                               .merge_neighbor_spans_by_box_coordinate()))

    assert 4 == len(MergeSpans(list_of_spans_to_merge, 0.04387334, 0.01421097).merge_neighbor_spans_by_box_coordinate())

    merge_spans = MergeSpans(list_of_spans_to_merge_2, 0.04387334, 0.01421097)
    assert 1 == len(merge_spans.merge_neighbor_spans_by_box_coordinate())

    assert [30113, 30359] == [merge_spans.merge_neighbor_spans_by_box_coordinate()[0].start, merge_spans.merge_neighbor_spans_by_box_coordinate()[0].end]


def test_merge_neighbor_spans_by_symbol_distance():
    assert 7 == (len(MergeSpans(list_of_spans_to_merge, index_distance=10)
                 .merge_neighbor_spans_by_symbol_distance()))


    assert 10 == len(MergeSpans(list_of_spans_to_merge, index_distance=1).merge_neighbor_spans_by_symbol_distance())

    list_of_spans_to_merge_2 = [
        Span(start=1, end=3, box=Box(l=0.1, t=0.2, w=0.2, h=0.2, page=11)),
        Span(start=5, end=7, box=Box(l=0.3, t=0.2, w=0.2, h=0.2, page=11)),
    ]

    merge_spans = MergeSpans(list_of_spans_to_merge_2, index_distance=1)
    result = merge_spans.merge_neighbor_spans_by_symbol_distance()
    assert 2 == len(result)

    assert set([(1, 3), (5, 7)]) == set([(entry.start, entry.end) for entry in result])

    merge_spans = MergeSpans(list_of_spans_to_merge_2, index_distance=4)
    result = merge_spans.merge_neighbor_spans_by_symbol_distance()
    assert 1 == len(result)

    assert set([(1, 7)]) == set([(entry.start, entry.end) for entry in result])
    assert [Box(l=0.1, t=0.2, w=0.4, h=0.2, page=11)] == [entry.box for entry in result]


def test_from_span_groups_with_box_groups():
    # convert test fixtures into SpanGroup with BoxGroup format
    list_of_spans_to_merge_in_span_group_format = []
    for span in list_of_spans_to_merge:
        list_of_spans_to_merge_in_span_group_format.append(
            SpanGroup(
                spans=[Span(start=span.start, end=span.end)],
                box_group=BoxGroup(boxes=[span.box])
            )
        )

    assert 7 == (len(MergeSpans.from_span_groups_with_box_groups(
                    list_of_spans_to_merge_in_span_group_format,
                    index_distance=10).merge_neighbor_spans_by_symbol_distance())
                 )

    assert len(list_of_spans_to_merge) == (len(MergeSpans.from_span_groups_with_box_groups(
        list_of_spans_to_merge_in_span_group_format,
        0,
        0).merge_neighbor_spans_by_box_coordinate()))


def test_box_groups_to_span_groups():
    with open(fixture_path / "20fdafb68d0e69d193527a9a1cbe64e7e69a3798__pdfplumber_doc.json", "r") as f:
        raw_json = f.read()
        fixture_doc_json = json.loads(raw_json)
        doc = Document.from_json(fixture_doc_json)

    with open(fixture_path / "20fdafb68d0e69d193527a9a1cbe64e7e69a3798__bib_entries.json", "r") as f:
        raw_json = f.read()
        fixture_bib_entries_json = json.loads(raw_json)["bib_entries"]

    box_groups = []
    # make box_groups from test fixture bib entry span groups (we will test the method to generate better spans)
    for bib_entry in fixture_bib_entries_json:
        box_groups.append(BoxGroup.from_json(bib_entry["box_group"]))

    overlap_span_groups = box_groups_to_span_groups(box_groups, doc) #, center=False)
    overlap_at_token_center_span_groups = box_groups_to_span_groups(box_groups, doc) #, center=True)

    assert (len(box_groups) == len(overlap_span_groups) == len(overlap_at_token_center_span_groups))

    # annotate both onto doc to extract texts:
    doc.annotate(overlap_span_groups=overlap_span_groups)
    doc.annotate(overlap_at_token_center_span_groups=overlap_at_token_center_span_groups)

    # when center=False, any token overlap with BoxGroup becomes part of the SpanGroup
    print(doc.overlap_span_groups[29].text)
    assert "[30]" in doc.overlap_span_groups[29].text
    assert "[31]" in doc.overlap_span_groups[29].text
    assert not doc.overlap_span_groups[29].text.startswith("[30]")
    assert not doc.overlap_span_groups[29].text.startswith("[30]")

    assert doc.overlap_at_token_center_span_groups[29].text.startswith("[30]")
    assert "[31]" not in doc.overlap_span_groups[29].text

    # SpanGroup span boxes are not saved, original box_group boxes are saved
    assert all([sg.spans[0].box is None for sg in doc.overlap_at_token_center_span_groups])
    assert all([sg.box_group is not None for sg in doc.overlap_at_token_center_span_groups])
