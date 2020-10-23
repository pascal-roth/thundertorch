import pandas as pd
from stfs_pytoolbox.ML_Utils import *
import numpy as np

import argparse
import os
import sys


def parseArguments():
    parser = argparse.ArgumentParser("This scripts plot a contour comparison between a pyTorch model prediction"
                                     " and a the original data and their difference using the"
                                     " pyFLUT.flut.Flut.contourplot functionality.")

    # Add mutually_exclusive_group to either load a FlexMLP model or create on based on input
    parser.add_argument('-m', '--models', dtype=str, nargs='+', required=True,
                        help='models that will be assembled into one model')
    parser.add_argument('--output', '-o', dest='output', required=False, default='AssemblyModel.pt',
                        help='file to which assembled model is saved in torch script format')
    parser.add_argument('--scale-flag', '-s', action="store_true", default=False, dest='scale',
                        help='enables limit_scale flag for Assembly model. It bounds model inputs to 0 and 1')

    return parser.parse_args()


def main():
    args = parseArguments()

    # create list to save scalar information
    ymin = []
    ymax = []
    models = []

    for model in args.models:
        model =
        model, features, labels, _, [featureScaler, labelScaler] = loadFlexMLPCheckpoint(file)

        ymin.append(labelScaler.data_min_)
        ymax.append(labelScaler.data_max_)
        models.append(model.eval())

    # feature scaler is only required once
    xmin = featureScaler.data_min_
    xmax = featureScaler.data_max_

    # Converting list into numpy ndarrays
    ymin = np.asarray(ymin)
    ymax = np.asarray(ymax)

    if args.scale:
        limit = False

    assemblyModel = AssemblyModel(models, xmin, xmax, ymin, ymax, limit_scale=args.scale)
    assemblyModel.toTorchScript(args.output)


if __name__ == "__main__":
    main()