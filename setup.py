import setuptools

setuptools.setup(
    name="mmda",
    version="0.0.2",
    python_requires=">= 3.7",
    packages=setuptools.find_packages(include=["mmda.*"]),
    install_requires=["intervaltree", "tqdm", "pdf2image", "pdfplumber", "requests"],
    extras_require={
        "dev": ["pytest"],
        "api": ["Flask", "gevent"],
        "pipeline": ["requests"],
        "lp_predictors": ["layoutparser",
                          "torch",
                          "torchvision",
                          "effdet"
                          ],
        "vila_predictors": ["vila",
                            "transformers"
                            ]
    },
)
