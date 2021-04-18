import numpy as np
import argparse

from thunder_torch import models


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("This scripts plot a contour comparison between a pyTorch model prediction"
                                     " and a the original data and their difference using the"
                                     " pyFLUT.flut.Flut.contourplot functionality.")

    # Add mutually_exclusive_group to either load a FlexMLP model or create on based on input
    parser.add_argument('-m', '--models', type=str, nargs='+', default=['./checkpoints/cpMean_64_64_softplus.ckpt',
                        './checkpoints/lambda_64_64_softplus.ckpt', './checkpoints/hMean_64_64_softplus.ckpt',
                        './checkpoints/rho_64_64_softplus.ckpt'], help='models that will be assembled into one model')
    parser.add_argument('-t', '--type', type=str, default='LightningFlexMLP',
                        help='type of the models that should be assembled')
    parser.add_argument('--output', '-o', dest='output', required=False, default='AssemblyModel.pt',
                        help='file to which assembled model is saved in torch script format')
    parser.add_argument('--scale-flag', '-s', action="store_true", default=False, dest='scale',
                        help='enables limit_scale flag for Assembly model. It bounds model inputs to 0 and 1')

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # create list to save scalar information
    ymin = []
    ymax = []
    model_list = []

    for model_path in args.models:
        print(model_path)
        model = getattr(models, args.type).load_from_checkpoint(model_path)

        ymin.append(model.hparams.lparams.y_scaler.data_min_)
        ymax.append(model.hparams.lparams.y_scaler.data_max_)
        model_list.append(model.eval())

    # feature scaler is only required once
    xmin = model.hparams.lparams.x_scaler.data_min_
    xmax = model.hparams.lparams.x_scaler.data_max_

    # Converting list into numpy ndarrays
    ymin = np.asarray(ymin)
    ymax = np.asarray(ymax)

    assemblyModel = models.AssemblyModel(model_list, xmin, xmax, ymin, ymax, limit_scale=args.scale)
    assemblyModel.toTorchScript(args.output)


if __name__ == "__main__":
    args = parseArguments()
    print(args)
    main(args)