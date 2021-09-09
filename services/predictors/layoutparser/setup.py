import os
import setuptools


DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_NAME = open(os.path.join(DIR, "PROJECT_NAME.txt"), "r").read().strip()

# DO NOT CHANGE: required by the s2agemaker template,
s2agemaker_requirements = [
    "gunicorn",
    "uvicorn[standard]",
    "pydantic",
    "fastapi",
    "click",
    "python-json-logger",
]

# Add your python dependencies
model_requirements = ["pdf2image", "intervaltree", "layoutparser"]

dev_requirements = ["pytest", "mypy", "black", "requests", "types-requests"]

setuptools.setup(
    name=PROJECT_NAME,
    version="0.0.1",
    description="Describe your model here",
    url="https://github.com/allenai/mmda/",
    packages=setuptools.find_packages(),
    install_requires=s2agemaker_requirements + model_requirements,
    extras_require={"dev": dev_requirements},
    python_requires="~= 3.8",
)
