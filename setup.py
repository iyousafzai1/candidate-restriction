#!/usr/bin/env python3
"""Setup script for candidate-restriction package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="candidate-restriction",
    version="1.0.0",
    author="Irfan Khan, Ramesh Kumar Ayyasamy, Miao Zhang",
    description="Adaptive candidate restriction for scalable random-forest Bayesian optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/candidate-restriction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
)
