import argparse

from Diffusion.Train import train, eval


def main(args, model_config = None):
    modelConfig = {
        "state": args.state,
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "../ckpts/",
        "test_load_weight": args.ckpt,
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": args.img_name,
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", "-s", choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--img_name", type=str, default="SampledImg.png")
    args = parser.parse_args()
    if args.state == "eval":
        for i in range(1, 11):
            main(args)
            args.img_name = f"SampledImg{i}.png"
    else:
        main(args)
