# setup.py
from setuptools import setup, find_packages

setup(
    name="pylis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "netCDF4",
        "pandas",
        "scipy",
        "tqdm",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "cartopy"
    ],
    python_requires=">=3.8",
    description="Python library to read and plot LIS model data",
    author="Vinayak Huggannavar",
    author_email="vinayak.huggannavar@kuleuven.be", 
    url="https://github.com/vinni94/pylis",  # optional
)
