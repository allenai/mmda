import setuptools

setuptools.setup(
    name="mmda",
    version="0.0.2",
    python_requires=">= 3.7",
    packages=setuptools.find_packages(include=["mmda.*"]),
    install_requires=["intervaltree", "pdf2image", "pdfplumber", "pandas", "requests"],
    extras_require={"dev": ["pytest"]},
)
