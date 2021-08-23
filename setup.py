#!/usr/bin/python
import setuptools

setuptools.setup(
    name='mmda',
    version='0.0.2',
    python_requires=">= 3.8",
    package_dir={"": "library"},
    packages=['mmda'],
    install_requires=['intervaltree', 'pdf2image'],
    extras_require={"dev": ["pytest"]}
)
