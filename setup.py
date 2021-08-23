import setuptools

setuptools.setup(
    name='mmda',
    version='0.0.2',
    python_requires=">= 3.8",
    packages=setuptools.find_packages(include=["mmda.*"]),
    install_requires=['intervaltree', 'pdf2image'],
    extras_require={"dev": ["pytest"]}
)
