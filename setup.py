# coding=utf-8
"""Setup script for the combined NAM + baseline project."""

from pathlib import Path

from setuptools import find_packages
from setuptools import setup


ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")


def build_package_list():
    packages = []
    for package in find_packages(include=["baseline", "baseline.*", "nam", "nam.*"]):
        if package == "nam" or package.startswith("nam."):
            packages.append(package.replace("nam", "neural_additive_models", 1))
        else:
            packages.append(package)
    packages.append("neural_additive_models")
    return sorted(set(packages))


INSTALL_REQUIRES = [
    "numpy>=1.24,<2",
    "pandas>=2.1",
    "scikit-learn>=1.3",
    "torch>=2.1",
    "matplotlib>=3.8",
    "tqdm>=4.66",
    "xgboost>=1.7",
    "interpret>=0.5",
    "tabulate>=0.9",
]


setup(
    name="hku-nam",
    version="0.1.0",
    description="Neural Additive Models with baseline comparison utilities.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="HKU NAM Project",
    python_requires=">=3.10",
    packages=build_package_list(),
    package_dir={"neural_additive_models": "nam"},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "nam-batch=main:main",
            "nam-compare=baseline.run_experiment:main",
            "nam-train=neural_additive_models.experiments.nam.train:main",
            "nam-evaluate=neural_additive_models.experiments.nam.evaluate:main",
            "nam-plot=neural_additive_models.experiments.nam.plot_ensemble:main",
            "nam-compas=neural_additive_models.experiments.compas_multitask.run:main",
            "nam-vs-fm=neural_additive_models.experiments.nam_vs_fm.run:main",
        ]
    },
)
