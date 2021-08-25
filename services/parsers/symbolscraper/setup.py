import setuptools


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
model_requirements = [
    "mmda@git+https://github.com/allenai/mmda.git@yogic/s2agemaker-symbol-scraper"
]

dev_requirements = ["pytest", "mypy", "black", "requests", "types-requests"]

setuptools.setup(
    name="mmda-symbolscraper",
    version="0.0.1",
    description="SymbolScraperParser as a Service",
    url="https://github.com/allenai/mmda/",
    packages=setuptools.find_packages(),
    install_requires=s2agemaker_requirements + model_requirements,
    extras_require={"dev": dev_requirements},
    python_requires="~= 3.8",
)
