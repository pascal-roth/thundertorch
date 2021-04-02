"""stfs_pytoolbox setup."""
import subprocess
import sys
from setuptools import find_packages, setup
import os
import versioneer

def install_with_pip(package):
        subprocess.call([sys.executable, "-m", "pip", "install", package])

def install_cantera_with_conda():
        os.popen("conda install -c cantera cantera==2.4").read()

install_cantera_with_conda()

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = find_packages()
packages_dir = {pack: pack for pack in packages}

setup(
    name="stfs_pytoolbox",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="a Toolbox with usefull functions for data analysis and postProcessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Hannes Böttler',
    author_email='boettler@stfs.tu-darmstadt.de',
    packages=packages,
    packages_dir=packages_dir,
    scripts=[
        'scripts/ML_Utils/assembleLightningModels',
        'scripts/ML_Utils/assemblyFlexNN',
        'scripts/ML_Utils/ml_init',
        'scripts/ML_Utils/trainFlexNN',
        'scripts/ML_Utils/trainFlexNNargs',
        'scripts/ML_Utils/trainFlexNNmulti'
        ],
    install_requires=[
        'numpy==1.16.4',
        'matplotlib==3.2.2',
        'scipy==1.4.1',
        'tabulate==0.8.7',
        'versioneer==0.18',
        'future==0.18.2',
        'deap==1.3.1',
        'autologging==1.3.2',
        'termcolor==1.1.0',
        'ruamel.yaml==0.16.12',
        'pyyaml==5.3.1',
        'mkl',
        'pytest==6.0.2',
        'pytest-mpl==0.11',
        'mock==4.0.2',
        'PyHamcrest==2.0.2',
        'pandas==1.1.2',
        'sphinx-pyreverse==0.0.13',
        'pylint==2.6.0',
        'ipython==7.16.1',
        'ipykernel',
        'jedi>=0.17.2',
        'gitpython',
        'tables==3.6.1',
        'pytest-dependency==0.5.1',
        'pytest-mypy==0.8.0',
        'nbstripout',
        'sphinx_rtd_theme'        
    ],
    extras_require={
        'pyTorch': ['torch==1.2.0',
                    'torchvision==0.4.0',
                    'scikit-learn==0.23.1',
                    'tqdm',
                    'dill',
                    'pytorch-lightning==0.7.6',
                    'comet-ml']
        }
    )
