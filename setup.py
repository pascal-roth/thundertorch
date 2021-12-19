"""ThunderTorch setup."""

import subprocess
import sys
from setuptools import find_packages, setup
import os
import versioneer


def install_with_pip(package) -> None:
    subprocess.call([sys.executable, "-m", "pip", "install", package])


with open("README.md", "r") as fh:
    long_description = fh.read()

packages = find_packages()
packages_dir = {pack: pack for pack in packages}

setup(
    name="thunder_torch",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="PyTorch and PyTorch Lightning wrapper for high-level AI research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Pascal Roth',
    author_email='roth.pascal@outlook.de',
    packages=packages,
    packages_dir=packages_dir,
    scripts=[
        'scripts/assembleLightningModels',
        'scripts/assemblyFlexNN',
        'scripts/ml_init',
        'scripts/trainFlexNN',
        'scripts/trainFlexNNmulti',
        'scripts/exportCppFlexNN'
        ],
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'tabulate',
        'versioneer',
        'future',
        'deap',
        'autologging',
        'termcolor',
        'ruamel.yaml',
        'pyyaml',
        'mkl',
        'pytest',
        'pytest-mpl',
        'mock',
        'PyHamcrest',
        'pandas',
        'sphinx-pyreverse',
        'pylint',
        'ipython',
        'ipykernel',
        'jedi',
        'gitpython',
        'tables',
        'pytest-dependency',
        'pytest-mypy',
        'nbstripout',
        'sphinx_rtd_theme',
        'docutils==0.17',
        'more_itertools'
    ],
    extras_require={
        'pyTorch': ['torch',
                    'torchvision',
                    'scikit-learn',
                    'tqdm',
                    'dill',
                    'pytorch-lightning']
        }
    )
