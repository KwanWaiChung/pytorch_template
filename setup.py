from setuptools import setup

setup(
    name="pytorch_template",
    version="1.0",
    description="Language processing library",
    author="Cyrus Kwan",
    author_email="15219666@life.hkbu.edu.hk",
    packages=["pytorch_template"],
    install_requires=[
        "torch==1.6.0",
        "tqdm==4.48.2",
        "matplotlib==3.3.0",
        "requests==2.24.0",
        "spacy==2.3.2",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz",
        "pandas==1.1.0",
        "scikit-learn==0.23.2",
        "tensorboard==2.3.0",
    ],
)
