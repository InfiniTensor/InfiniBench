import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from toolkit.tools import (
    dynamic_import,
    infer,
    load_model,
    load_model_config,
    load_tokenizer,
    load_yaml_config,
    parse_args,
)


def main():
    # parse input arguments
    args = parse_args()
    model = args.model
    model_type = model.split("-", 1)[0]
    prompt = args.prompt

    # load yaml config
    yaml_path = os.path.join("config", model_type, f"{model}.yaml")
    config = load_yaml_config(yaml_path)

    # load model, tokenizer
    model_config = load_model_config(config["model"]["config_path"])
    ModelClass = dynamic_import(config["model"]["class_path"])
    model = load_model(
        ModelClass,
        checkpoint_paths=config["weights"]["checkpoint_paths"],
        config=model_config,
    )
    tokenizer = load_tokenizer(config["tokenizer"]["path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.device = device
    model = model.to(device)

    infer(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
