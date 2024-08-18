import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res, _, _ = model(image)
        res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)