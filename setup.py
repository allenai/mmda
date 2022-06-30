import setuptools

setuptools.setup(
    name="mmda",
    version="0.0.11",
    python_requires=">= 3.7",
    packages=setuptools.find_packages(include=["mmda*", "ai2_internal*"]),
    install_requires=[
        "intervaltree",
        "tqdm",
        "pdf2image",
        "pdfplumber",
        "requests",
        "pandas",
        "pydantic"
    ],
    extras_require={
        "dev": ["pytest"],
        "api": ["Flask", "gevent"],
        "pipeline": ["requests"],
        "lp_predictors": ["layoutparser", "torch", "torchvision", "effdet"],
        "vila_predictors": ["vila >= 0.3.0", "transformers"],
        "bibentry_predictor": ["transformers", "unidecode", "torch"],
    },
    include_package_data=True,
)
