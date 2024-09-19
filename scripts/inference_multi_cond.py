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
from ldm.data.mul_cond import KvasirSegTest
from PIL import Image

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
def preprocess_mask(img):
    mask = np.zeros_like(img)
    mask[img >= 150] = 1
    return mask
def ldm_cond_sample_dataset(config_path, ckpt_path, dataset, batch_size,text,mask_only,text_only):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x = next(iter(dataloader))
    if mask_only:
        text = batch_size * ["a polyp"]
        seg = x['conditions']['segmentation']
    elif text_only:
        black_mask = np.zeros((batch_size, 2, 256, 256), dtype=np.uint8)
        text = batch_size * [text.lower()]  # 修正为列表
        seg = black_mask
        if not isinstance(seg, torch.Tensor):
                seg = torch.tensor(seg).to('cuda').float()
    else:
        seg = x['conditions']['segmentation']
        text = batch_size * [text.lower()]  # 修正为列表

    conditions = {
        'segmentation': seg,
        'text_data': text
    }
    real = x['image']
    real = rearrange(real, 'b h w c -> b c h w')


    with torch.no_grad():
        
        # seg = rearrange(seg, 'b h w c -> b c h w')
        
        condition = model.to_rgb(seg)
        print(condition)

        unconditions = {
        'segmentation': np.zeros((batch_size, 2, 256, 256), dtype=np.uint8),
        'text_data': batch_size * [""]
    }
        cond = model.get_learned_conditioning(conditions)
        uc = model.get_learned_conditioning(unconditions)
        print("cond:",cond.shape)
        
        samples, _ = model.sample_log(cond=cond, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, unconditional_guidance_scale=7.5,
                                                         unconditional_conditioning=uc,
                                                         
                                                         eta=0.0)

        samples = model.decode_first_stage(samples)
    results_dir = 'results/test'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_image((real+1.0)/2.0, os.path.join(results_dir, 'real.png'))
    save_image((samples+1.0)/2.0, os.path.join(results_dir, 'fake.png'))
    save_image(condition, os.path.join(results_dir, 'cond.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/multi_cond.yaml')
    parser.add_argument('--ckpt_path', type=str, default='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/3_4/2024-03-05T10-34-43_multi_cond/checkpoints/trainstep_checkpoints/epoch=76-step=24999.ckpt')
    parser.add_argument('--batch_size', type=str, default=4)
    parser.add_argument("--text",type=str,default="a red polyp",help="text condition")
    parser.add_argument("--mask_only",type=bool,default=False,help="if mask only")
    parser.add_argument("--text_only",type=bool,default=True,help="if text only")
    args = parser.parse_args()

    dataset = KvasirSegTest(size=256)
    ldm_cond_sample_dataset(args.config_path, args.ckpt_path, dataset, args.batch_size,args.text,args.mask_only,args.text_only)
