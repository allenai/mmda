import setuptools

setuptools.setup(
    name="mmda",
    description="mmda",
    version="0.0.17",
    url="https://www.github.com/allenai/mmda",
    python_requires=">= 3.7",
    packages=setuptools.find_packages(include=["mmda*", "ai2_internal*"]),
    install_requires=[
        "tqdm",
        "pdf2image",
        "pdfplumber",
        "requests",
        "pandas",
        "pydantic",
        "ncls",
    ],
    extras_require={
        "dev": ["pytest"],
        "api": ["Flask", "gevent"],
        "pipeline": ["requests"],
        "spacy_predictors": ["spacy"],
        "lp_predictors": ["layoutparser", "torch", "torchvision", "effdet"],
        "vila_predictors": ["vila>=0.4.2,<0.5", "transformers"],
        "bibentry_predictor": ["transformers", "unidecode", "torch"],
    },
    include_package_data=True,
)
