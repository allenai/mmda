from setuptools import find_namespace_packages, setup

setup(
    name="mmda",
    description="mmda",
    version="0.0.22",
    url="https://www.github.com/allenai/mmda",
    python_requires=">= 3.7",
    packages=find_namespace_packages(include=["mmda*", "ai2_internal*"]),
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
        "mention_predictor": ["torch==1.12.0", "transformers==4.20.1", "optimum[onnxruntime-gpu]"],
        "bibentry_predictor": ["transformers", "unidecode", "torch"],
        "citation_links": ["numpy", "thefuzz[speedup]", "sklearn", "xgboost"],
    },
    include_package_data=True,
    package_data={
        "ai2_internal.citation_mentions.data": ["*"],
        "ai2_internal.vila.test_fixtures": ["*"],
        "ai2_internal.shared_test_fixtures": ["*"]
    }
)
