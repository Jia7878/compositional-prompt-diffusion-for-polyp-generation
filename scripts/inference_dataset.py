import torch
import numpy as np
import argparse
import os

# from sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.data.kvasir import KvasirSegTrain, KvasirSegEval

# NOTE: You have to be inside latent-diffusion folder in order to run the script
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def ldm_cond_sample_dataset(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x = next(iter(dataloader))

    real = x['image']
    print("real",real.shape)
    real = rearrange(real, 'b h w c -> b c h w')
    print("real_",real.shape)
    seg = x['segmentation']
    print("seg",seg.shape)

    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)
        print("condition",condition.shape)

        print(condition)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, eta=1.)

        samples = model.decode_first_stage(samples)
    results_dir = 'results/test'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_image((real+1.0)/2.0, os.path.join(results_dir, 'real.png'))
    save_image((samples+1.0)/2.0, os.path.join(results_dir, 'fake.png'))
    save_image(condition, os.path.join(results_dir, 'cond.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='latent-diffusion/models/ldm/semantic_synthesis256/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default='/home/hyl/yujia/checkpoint/continue/checkpoints/last.ckpt')
    parser.add_argument('--batch_size', type=str, default=4)
    args = parser.parse_args()

    dataset = KvasirSegEval(size=256)
    ldm_cond_sample_dataset(args.config_path, args.ckpt_path, dataset, args.batch_size)
