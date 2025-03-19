from setuptools import setup, find_packages

setup(
    name="gifflar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch_geometric>=2.6.1",
        "rdkit>=2022",
        "scikit-learn",
        "numpy",
        "pandas",
        "glyles>=1.0.0",
        "glycowork @ git+https://github.com/BojarLab/glycowork",
        "jsonargparse",
        "rich",
        "pytorch-lightning",
        "pytest",
        "pyyaml",
        "networkx",
        "torchmetrics",
        "transformers",
        "sentencepiece",
        "xformers==0.0.28.post1",
        "protobuf",
        "torch_scatter"
    ],
    description="Package for using BojarLab's GIFFLAR models (package is not assoicated with BojarLab)",
    url="https://github.com/loganwoudstra/GIFFLAR",
)