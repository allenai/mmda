from setuptools import find_namespace_packages, setup

setup(
    name="mmda",
    description="mmda",
    version="0.0.30",
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
        "hf_predictors": ["torch", "transformers", "smashed@git+https://github.com/allenai/smashed@v0.0.1"],
        "vila_predictors": ["vila>=0.4.2,<0.5", "transformers"],
        "mention_predictor": ["transformers[torch]", "optimum[onnxruntime]"],
        "mention_predictor_gpu": ["transformers[torch]", "optimum[onnxruntime-gpu]"],
        "bibentry_predictor": ["transformers", "unidecode", "torch"],
        "bibentry_detection_predictor": ["layoutparser", "torch==1.8.0", "torchvision==0.9.0"],
        "citation_links": ["numpy", "thefuzz[speedup]", "sklearn", "xgboost"],
    },
    include_package_data=True,
    package_data={
        "ai2_internal.bibentry_detection_predictor.data": ["*"],
        "ai2_internal.bibentry_predictor_mmda.data": ["*"],
        "ai2_internal.citation_mentions.data": ["*"],
        "ai2_internal.vila.test_fixtures": ["*"],
        "ai2_internal.shared_test_fixtures": ["*"]
    }
)
