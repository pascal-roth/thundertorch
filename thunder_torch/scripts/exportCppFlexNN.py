import torch
import argparse
from thunder_torch import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-mp', '--model_path', type=str,
                        help='Model name')
    parser.add_argument('-mt', '--model_type', type=str, default='LightningFlexMLP',
                        help='Model type')
    parser.add_argument('-r', '--output_relu', action='store_true',
                        help="Adds a relu activation function to output layer")

    return parser.parse_args()


def export_model(args: argparse.Namespace) -> None:
    model = getattr(models, args.model_type).load_from_checkpoint(args.model)
    n_inp = model.hparams.n_inp  # TODO: how for all models?
    example = torch.rand(n_inp)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save("traced_model.pt")

if __name__ == '__main__':
    args = parse_args()
    export_model(args)


