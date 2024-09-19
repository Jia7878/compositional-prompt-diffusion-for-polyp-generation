import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
import json

def preprocess_mask(img):
    mask = np.zeros_like(img)
    mask[img >= 150] = 1
    return mask

class SegmentationBase(Dataset):
    def __init__(self,
                 image_mask_csv, image_text_json, data_root, segmentation_root, text_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=2, shift_segmentation=False,
                 conditions=['segmentation', 'text_data'],
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.image_mask_csv = image_mask_csv
        self.image_text_json = image_text_json
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        self.text_root = text_root
        self.conditions = conditions
        
        # Load image-mask data
        with open(self.image_mask_csv, "r") as f:
            self.image_paths = f.read().splitlines()

        with open(self.image_text_json, "r") as f:
            self.image_text_data = json.load(f)

        self._length = len(self.image_paths) + len(self.image_text_data)
        self.labels = []

        for image_name in self.image_paths:
            image_path = os.path.join(self.data_root, image_name)
            mask_name = os.path.splitext(image_name)[0] + ".jpg"
            mask_path = os.path.join(self.segmentation_root, mask_name)
            self.labels.append({"image_path": image_path,
                                "segmentation_path": mask_path,
                                "text_data": ""})  # Initialize text data to "a polyp"
        
        # Load image-text data
        for image_text_name, text_data in self.image_text_data.items():
            image_text_path = os.path.join(self.text_root, image_text_name)
            text = text_data.get("Beard_and_Age", "").lower()
            self.labels.append({"image_path": image_text_path,
                                "segmentation_path": "",
                                "text_data": text})

        size = None if size is not None and size <= 0 else size
        self.size = size

        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4
            }[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = self.labels[i]
        # Read image
        image = Image.open(example["image_path"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        # Resize image if required
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        
        # Read segmentation mask if exists
        if example["segmentation_path"] is not None:
            segmentation = Image.open(example["segmentation_path"])
            if not segmentation.mode == "L":
                segmentation = segmentation.convert("L")
            assert segmentation.mode == "L", segmentation.mode
            segmentation = np.array(segmentation).astype(np.uint8)
            # Preprocess segmentation mask
            segmentation = preprocess_mask(segmentation)
            if self.shift_segmentation:
                segmentation = segmentation + 1
            if self.size is not None:
                segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        else:
            segmentation = None  # If segmentation path is None, set segmentation to None
        
        # Get text data
        text_data = example.get("text_data")
        
        # Apply preprocessing to image
        if segmentation is not None:
            if self.size is not None:
                processed = self.preprocessor(image=image,
                                            mask=segmentation)
            else:
                processed = {"image": image,
                            "mask": segmentation}
        
        # Normalize image and prepare segmentation mask as one-hot encoding
        example["image"] = (processed["image"] / 127.5 - 1.0).astype(np.float32)
        if segmentation is not None:
            segmentation = processed["mask"]
            onehot = np.eye(self.n_labels)[segmentation]
            onehot_permuted = np.transpose(onehot, (2, 0, 1))
            example["segmentation"] = onehot_permuted.astype(np.float32)
        else:
            example["segmentation"] = None  # Set segmentation to None if it doesn't exist
        
        example["text_data"] = text_data  # Add text data to the example
        
        # Prepare condition dictionary
        condition_dict = {}
        for condition_name in self.conditions:
            if condition_name == "segmentation":
                condition_dict[condition_name] = example["segmentation"]
            elif condition_name == "text_data":
                condition_dict[condition_name] = example["text_data"]
        
        return {"image": example["image"], "conditions": condition_dict}


class KvasirSegTest(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(image_mask_csv='data/kvasir/kvasir_eval.txt',
                         data_root='data/kvasir/images',
                         segmentation_root='data/kvasir/masks',
                         text_root='/home/hyl/yujia/data_Description/test/images',
                         image_text_json='/home/hyl/yujia/data_Description/test/images/metadata.json',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)
class multi_cond_val(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(
            image_mask_csv='data/kvasir/kvasir_train.txt',  # 图片-掩模对的文件
            image_text_json='/home/hyl/yujia/data_Description/test/images/metadata.json',  # 图片-文本对的JSON文件
            data_root='data/kvasir/images',
            segmentation_root='data/kvasir/masks',
            text_root='/home/hyl/yujia/data_Description/test/images',  # 文本文件的根目录
            size=size, random_crop=random_crop, interpolation=interpolation,
            n_labels=2
        )
class multi_cond_train(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(
            image_mask_csv='/home/hyl/yujia/data_5000_384/train.txt',  # 图片-掩模对的文件
            image_text_json='/home/hyl/yujia/data_Description/train/images_256/metadata.json',  # 图片-文本对的JSON文件
            data_root='/home/hyl/yujia/data_5000_384/images_256',
            segmentation_root='/home/hyl/yujia/data_5000_384/masks_grey_256',
            text_root='/home/hyl/yujia/data_Description/train/images_256',  # 文本文件的根目录
            size=size, random_crop=random_crop, interpolation=interpolation,
            n_labels=2
        )



def write_lines(file, lines):
    with open(file, 'w') as f:
        for line in lines:
            f.write(os.path.basename(line))
            f.write('\n')


def generateKvasirCSV(dir, output, train=0.9):
    files = glob.glob(dir)
    random.shuffle(files)
    length = len(files)

    train_data = files[:int(train * length)]
    write_lines(f'{output}/kvasir_train.txt', train_data)

    eval_data = files[int(train * length):]
    write_lines(f'{output}/kvasir_eval.txt', eval_data)
