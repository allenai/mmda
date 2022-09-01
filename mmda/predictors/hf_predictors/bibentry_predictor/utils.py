from typing import List

from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.types.span import Span
from mmda.predictors.hf_predictors.bibentry_predictor.types import BibEntryPredictionWithSpan, BibEntryStructureSpanGroups


_SPAN_JOINER = " "


def mk_bib_entry_strings(document: Document) -> List[str]:
    return [_SPAN_JOINER.join(bib_entry.symbols) for bib_entry in document.bib_entry_boxes]


def map_raw_predictions_to_mmda(
        bib_entries: List[SpanGroup],
        raw_preds: List[BibEntryPredictionWithSpan]
) -> BibEntryStructureSpanGroups:
    """
    Fussy, and hopefully short-lived logic that can map the spans predicted for a given
    bib entry string back into its corresponding MMDA Document.

    Underlying predictor operates over bib entry strings, requiring us to:

    1. build each string from one or more mmda spans (which may not be contiguous [or even nearby])
    2. map the span for each inferred bib entry component back into original document, using
        one or more `mmda.types.span.Span`s
    """
    prediction = BibEntryStructureSpanGroups()

    for i in range(len(bib_entries)):
        sg = bib_entries[i]
        raw_pred = raw_preds[i]

        # Build up relative positions for each SG span within the SG text
        intervals = []
        curr_total = 0
        for span in sg.spans:
            i_start = curr_total
            curr_total += span.end - span.start
            intervals.append((i_start, curr_total))
            curr_total += len(_SPAN_JOINER)

        def map_raw_span(target, raw_span):
            if not raw_span:
                return

            new_spans = []

            cur_pos = raw_span.start
            end = raw_span.end

            for index, (interval_start, interval_end) in enumerate(intervals):
                sg_span = sg.spans[index]

                if interval_start <= cur_pos < interval_end:
                    new_sg_span_start = sg_span.start + (cur_pos - interval_start)

                    if end >= interval_end:
                        # Continues into next span, or ends at exact end of this one.
                        new_sg_span_end = sg_span.end
                    else:
                        # Ends in the middle of this span.
                        new_sg_span_end = new_sg_span_start + end - cur_pos

                    new_spans.append(Span(start=new_sg_span_start, end=new_sg_span_end))

                    # Advance our current position, accounting for whitespace before beginning of next span.
                    cur_pos = interval_end + len(_SPAN_JOINER)
                    if cur_pos >= end:
                        break

            target.append(SpanGroup(spans=new_spans))

        map_raw_span(prediction.bib_entry_number, raw_pred.citation_number)
        for author in (raw_pred.authors or []):
            map_raw_span(prediction.bib_entry_authors, author)
        map_raw_span(prediction.bib_entry_title, raw_pred.title)
        map_raw_span(prediction.bib_entry_venue_or_event, raw_pred.journal_venue_or_event)
        map_raw_span(prediction.bib_entry_year, raw_pred.year)
        map_raw_span(prediction.bib_entry_doi, raw_pred.doi)
        map_raw_span(prediction.bib_entry_url, raw_pred.url)

    return prediction

