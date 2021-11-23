# Extract reference data using a combination of methods

## Approach overview

1. Parse and rasterize a PDF document
1. Extract blocks using block model that is agnostic to anything other than text blocks
1. Apply a token predictor (i.e., VILA) using the step 2 blocks, blocks from layout parser or no blocks if not desired
1. Extract all tokens using the token predictions from step 3
1. For each bibliography token add to the block from step 2 that contains the highest fraction of the token's box area
1. For each block-with-biblio-tokens join the text of the reference together and ask Grobid to "processCitation" to get the paper title

Note that the Grobid predictions for author names seem unreliable in the default configuration. This example code focuses on paper title instead.

Specifically in the example code here:

* Tesseract as the block predictor -- This correctly puts each reference into a standalone block in basic tests
* HVILA (grotoap2 fine-tuned) for token prediction simply feeding Tesseract blocks -- Unfortunately only the tokens on the first page of References are classified as bibliography. The second page is excluded from results as the tokens are classified differently (perhaps due to lack of "References" header).

Finally, we use max box overlap as box containment didn't have as good of results on basic tests.

Running the example script here results in (example output)[https://github.com/allenai/mmda/pull/64] for the (MAML paper)[https://arxiv.org/pdf/1703.03400.pdf] (see page 9 of the PDF). Line 1 is the full reference text, line 2 is the extracted title. The results are pretty accurate for that page. Obviously this is non-conclusive at scale, however, the method does not require tuning and so may generalize well assuming good token classification and box prediction (problems on which we have good results already).

Ideally the extracted titles are enough to match to a publisher-provided metadata title (i.e., cleaner title) on S2. 

## Licensing notes

This example uses (Tesseract OCR)[https://github.com/tesseract-ocr/tesseract] for some text block prediction. The project is released under an Apache 2.0 license as open source. Default models for many character sets are provided as part of the project.
