import argparse
import json
from os import path

import torch

from eipl.model import SARNN

def load_config(file):
    with open(file, "r") as f:
        config = json.load(f)

    #! temp fix
    # joint_dim is 5 for open_manipulator
    # im_size is 64x64 for object grasp task
    # hard coded for now
    config["joint_dim"] = 5
    config["im_size"] = [64, 64]

    return config


def export_onnx(weight, output, config, verbose=False):
    model = SARNN(
        rec_dim=config["rec_dim"],
        joint_dim=config["joint_dim"],
        k_dim=config["k_dim"],
        heatmap_size=config["heatmap_size"],
        temperature=config["temperature"],
        im_size=config["im_size"],
    )

    ckpt = torch.load(weight, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_names = ["i.image", "i.joint", "i.state_h", "i.state_c"]
    output_names = ["o.image", "o.joint", "o.enc_pts", "o.dec_pts", "o.state_h", "o.state_c"]
    dummy_input = (
        torch.randn(1, 3, config["im_size"][0], config["im_size"][1]),
        torch.randn(1, config["joint_dim"]),
        tuple(torch.randn(1, config["rec_dim"]) for _ in range(2)),
    )
    torch.onnx.export(
        model,
        dummy_input,
        output,
        input_names=input_names,
        output_names=output_names,
        verbose=verbose,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight", help="PyTorch model weight file")
    parser.add_argument("--config", help="Configuration file", default=None)
    parser.add_argument("--output", help="Output ONNX file", default="model.onnx")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
    else:
        config_path = path.join(path.dirname(args.weight), "args.json")
        config = load_config(config_path)

    export_onnx(args.weight, args.output, config, args.verbose)
