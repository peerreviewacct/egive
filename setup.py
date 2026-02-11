from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="egive",
    version="0.1.0",
    author="Anonymous for peer review",  
    author_email="",  
    description="A Python package for EGIVE, an efficient variable importance and interaction detection method for black-box ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peerreviewacct/egive", 
    packages=find_packages(),
    classifiers=[
    "License :: OSI Approved :: Apache Software License", 
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=2.0.2, < 3",
        "pandas >= 1.5.3, < 3",
        "scipy>=1.6.3, < 2",
        "joblib>=1.5.3, < 2",
        "altair>=6.0.0",
        "scikit-learn>=1.6.1, < 2",
        "statsmodels>=0.14.6, < 1",
        "vegafusion[embed] >= 1.5.0",
       "vl-convert-python >= 1.6.0",
        "pyarrow >= 23.0.0, < 24"
    ],
)
