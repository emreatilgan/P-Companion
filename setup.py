# setup.py
from setuptools import setup, find_packages

setup(
    name="p_companion",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "tensorboard",
        "pytest"
    ],
)