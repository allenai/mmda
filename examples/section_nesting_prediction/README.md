# Use VILA + XGBoost for section extraction and nesting prediction

First we'll use VILA to predict section tokens then follow-up with a nesting prediction for each section. An example PDF and trained model are provided in the directory.

## Installation 

From within the project root directory

1. Install the necessary python dependencies

    ```bash
    pip install -e "[.section_nesting]"
    pip install -r ./examples/section_nesting_prediction/requirements.txt
    ```

2. Install poppler for rendering PDF images - the installation methods are different based on your platform:

    - Mac: `brew install poppler`
    - Ubuntu: `sudo apt-get install -y poppler-utils`
    - Windows: See [this post](https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows)

## Usage

from the project root directory

```bash
python examples/section_nesting_prediction/main.py
```

For the sample PDF you will see output as follows:

```text
...

Section 'I. INTRODUCTION' is top-level!
Section 'II. RELATED WORKS' is top-level!
Section 'III. PROBLEM FORMULATION' is top-level!
Section 'IV. LRCO: LEARNING FOR ROBUST COMBINATORIAL OPTIMIZATION' is top-level!
Section 'A. Overview' has parent 'IV. LRCO: LEARNING FOR ROBUST COMBINATORIAL OPTIMIZATION'
Section 'B. Maximizer Network' has parent 'IV. LRCO: LEARNING FOR ROBUST COMBINATORIAL OPTIMIZATION'
Section 'C. Minimizer Network' has parent 'IV. LRCO: LEARNING FOR ROBUST COMBINATORIAL OPTIMIZATION'
Section 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING' is top-level!
Section 'A. Background and Problem Formulation' has parent 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING'
Section 'B. Simulation Setup' has parent 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING'
Section 'D. Results with A Large Error Budget' has parent 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING'
Section 'E. Results with A Small Error Budget' has parent 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING'
Section 'F. Sensitivity Study' has parent 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING'
Section 'G. Inference Time' has parent 'V. APPLICATION: TASK OFFLOADING IN VEHICULAR EDGE COMPUTING'
Section 'VI. CONCLUSION' is top-level!
Section 'ACKNOWLEDGEMENT' is top-level!
Section 'REFERENCES' is top-level!
Section 'APPENDIX CONTEXT PARAMETER PREDICTION' is top-level!
```