# compositional-prompt-diffusion-for-polyp-generation
## Requirements



```bash
conda env create -f environment.yaml
conda activate ldm
```
## Train your own models

```bash
python  main.py \
--logdir 'resume_text_mask' \
--base 'configs/latent-diffusion/re_sd_mask.yaml' \
-t  --gpus 0,
```
## Dataset Polyplus 
Our dataset can be found here [Polyplus](https://drive.google.com/file/d/1TUeOOZhgbvw5sNaJFJ9x-wo1LSMKIc4o/view?usp=sharing) and [Benign&Malignant](https://drive.google.com/drive/folders/1l8z4roLsDSWGvEsqZMGkdnbp5t3U44bm?usp=drive_link)
