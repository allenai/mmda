import setuptools

setuptools.setup(
    name="mmda",
    description="mmda",
    version="0.0.19",
    url="https://www.github.com/allenai/mmda",
    python_requires=">= 3.7",
    packages=setuptools.find_packages(include=["mmda*", "ai2_internal*"]),
    install_requires=[
        "tqdm",
        "pdf2image",
        "pdfplumber@git+https://github.com/allenai/pdfplumber@63db31f8452c93d72baaca1e843f2ab68bc6ca85",
        "requests",
        "pandas",
        "pydantic",
        "ncls",
    ],
    extras_require={
        "dev": ["pytest"],
        "spacy_predictors": ["spacy"],
        "lp_predictors": ["layoutparser", "torch", "torchvision", "effdet"],
        "vila_predictors": ["vila>=0.4.2,<0.5", "transformers"],
        "mentions": ["transformers[torch]"],
        "bibentry_predictor": ["transformers", "unidecode", "torch"],
        "bibentry_detection_predictor":["layoutparser", "torch==1.8.0", "torchvision==0.9.0", "detectron2"]
    },
    include_package_data=True,
)
