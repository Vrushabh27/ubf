from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ubf",
    version="0.1.0",
    author="Vrushabh Zinage, Efstathios Bakolas",
    author_email="vrushabh.zinage@gmail.com",
    description="Universal Barrier Function (UBF) - A framework for implementing UBF methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vrushabbh27/ubf",
    project_urls={
        "Bug Tracker": "https://github.com/Vrushabbh27/ubf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "cvxpy>=1.1.0",
        "scipy>=1.5.0",
    ],
    test_suite="tests",
) 