# MMDA - multimodal document analysis

This is work in progress... 

## Setup

```bash
conda create -n mmda python=3.8
pip install -e '.[dev,<extras_require section from setup.py>]'
```

## Unit testing
Note that pytest is running coverage, which checks the unit test coverage of the code.
The percent coverage can be found in setup.cfg file.
```bash
pytest
```
for latest failed test
```bash
pytest --lf --no-cov -n0
```
for specific test name of class name
```bash
pytest -k 'TestFigureCaptionPredictor' --no-cov -n0
```

## Quick start

TBD