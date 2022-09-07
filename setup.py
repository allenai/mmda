from setuptools import find_namespace_packages, setup

setup(
    name="mmda",
    description="mmda",
    version="0.0.33",
    url="https://www.github.com/allenai/mmda",
    python_requires=">= 3.7",
    packages=find_namespace_packages(include=["mmda*", "ai2_internal*"]),
    install_requires=[
        "tqdm",
        "pdf2image",
        "pdfplumber>0.7.1",
        "requests",
        "pandas",
        "pydantic",
        "ncls",
    ],
    extras_require={
        "dev": ["pytest"],
        "spacy_predictors": ["spacy"],
        "lp_predictors": ["layoutparser", "torch", "torchvision", "effdet"],
        "hf_predictors": ["torch", "transformers", "smashed==0.1.10"],
        "vila_predictors": ["vila>=0.4.2,<0.5", "transformers"],
        "mention_predictor": ["transformers[torch]", "optimum[onnxruntime]"],
        "mention_predictor_gpu": ["transformers[torch]", "optimum[onnxruntime-gpu]"],
        "bibentry_predictor": ["transformers", "unidecode", "torch", "optimum[onnxruntime]"],
        "bibentry_predictor_gpu": ["transformers", "unidecode", "torch", "optimum[onnxruntime-gpu]"],
        "bibentry_detection_predictor": ["layoutparser", "torch==1.8.0+cu111", "torchvision==0.9.0+cu111"],
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
