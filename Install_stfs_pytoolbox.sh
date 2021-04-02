#!/bin/bash

auto="False"
### Introduce options to the bash script ###
while [ -n "$1" ]; do # while loop starts

    case "$1" in

    -a) echo "automated build for testing"; auto="True" ;; # Message for -a option
    *) echo "Option $1 not recognized"; echo "Use"; echo "-a for automated testing (no user input)"; exit;; # In case you typed a different option other than a,b,c

    esac

    shift

done

echo "Build without user-input: $auto"


echo "+-----------------------------------------+"
echo "|    Installing stfs_pytoolbox            |"
echo "+-----------------------------------------+"

if [[ $auto == "False" ]]; then
    read -p "Conda path [$HOME/miniconda3]: " condaPath
    read -p "name of conda environment where toolbox should be installed [stfs_pytoolbox]: " condaEnvName
    read -p "Please enter author name for RDM ["$USER"]: " author
    read -p "also install pyTorch functionality features? [y|n]: " ptFLAG
fi

condaPath=${condaPath:-$HOME/miniconda3}
condaEnvName=${condaEnvName:-stfs_pytoolbox}
author=${author:-$USER}

echo $author > stfs_pytoolbox/RDM/author.txt

echo ""
echo "Conda setup for installation:"
echo "Conda path: $condaPath, ENV-name: $condaEnvName"
echo "+-----------------------------------------+"

source ${condaPath}/etc/profile.d/conda.sh
source ${condaPath}/bin/activate
env_exist=$(conda info --envs | grep $condaEnvName)

if [ ! -z "$env_exist" ]
then
  echo "Activate existing conda environment"
  conda activate $condaEnvName
else
  echo "Create new conda environment"
  # Create conda environment
  conda create --name $condaEnvName python=3.6 pip=20.2.3 -y
  conda activate $condaEnvName
fi

# install package with pip
if [[ $ptFLAG == "y" ]] || [[ $auto == "True" ]]; then
    # install with machine learning
    echo "Installing stfs_pythonToolBox with pyTorch support"
    pip install -e .[pyTorch]
else
    # install package without machine learning functionality (pyTorch)
    echo "Installing stfs_pythonToolBox without pyTorch support"
    pip install -e .
fi

if [[ $auto == "False" ]]; then
    read -p "Install jupyter kernel? [y|n]" kernelFLAG
    if [[ $ptFLAG == "y" ]]; then
        ipython kernel install --user --name=$condaEnvName
    fi
fi

conda deactivate

echo ">>> Installation completed"

