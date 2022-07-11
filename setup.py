import setuptools

setuptools.setup(
    name="mmda",
    version="0.0.14",
    python_requires=">= 3.7",
    packages=setuptools.find_packages(include=["mmda*", "ai2_internal*"]),
    install_requires=[
        "intervaltree",
        "tqdm",
        "pdf2image",
        "pdfplumber@git+https://github.com/allenai/pdfplumber@63db31f8452c93d72baaca1e843f2ab68bc6ca85",
        "requests",
        "pandas",
        "pydantic"
    ],
    extras_require={
        "dev": ["pytest"],
        "spacy_predictors": ["spacy"],
        "lp_predictors": ["layoutparser", "torch", "torchvision", "effdet"],
        "vila_predictors": ["vila>=0.4.2,<0.5", "transformers"],
        "mentions": ["transformers[torch]"],
        "bibentry_predictor": ["transformers", "unidecode", "torch"],
    },
    include_package_data=True,
)
