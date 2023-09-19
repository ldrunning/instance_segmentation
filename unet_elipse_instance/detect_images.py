import os

import cv2
import numpy as np
import torch
from torch.utils import data as data_

from model.unet import UNet
from data import SingleImage, list_file_tree, SlideDataSet
from tqdm import tqdm
from utils import top_regions
from PIL import Image


def detect_slide(path, dir_root, model_path):
    unet = UNet(3, 3).cuda()
    unet.load_state_dict(torch.load(model_path))
    unet.eval()
    unet = torch.nn.DataParallel(unet).cuda()
    dataset = SlideDataSet(path, level=0, step=(640, 640),
                           read_size=(700, 700), out_size=(256, 256))

    # dataset = SingleImage(path)
    dataloader = data_.DataLoader(dataset,
                                  batch_size=6,
                                  num_workers=4,
                                  drop_last=False)
    # file_list = dataset.image_list
    dirname, file = os.path.split(path)
    print(path)
    save_path = dirname.replace("/media/khtao/my_book/openslide_data/original_data",
                                dir_root)
    save_path = os.path.join(save_path, file[:-4])
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)
    # top_regions(dataset, save_path, 10)
    softmax = torch.nn.Softmax2d().cuda()
    num = 0
    for imgs, position in tqdm(dataloader):
        imgs = imgs.cuda()
        masks = softmax(unet(imgs)[0])
        # masks = masks.detach().cpu().numpy()
        for mask, img, pos in zip(masks, imgs, position):
            # file_path = file_list[num]
            if torch.sum(mask[2, :, :] > 0.5) > 100:
                mask = mask.detach().cpu().numpy()
                img = img.detach().cpu().numpy()
                mask = (mask.transpose((1, 2, 0))) * 255
                img = (img.transpose((1, 2, 0))) * 255
                filename = str(int(pos[0][0])) + "_" + str(int(pos[0][1])) + ".png"
                cv2.imwrite(os.path.join(save_path, "masks", filename), mask.astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, "images", filename), img.astype(np.uint8)[:, :, ::-1])
            num += 1
    top_regions(dataset, save_path, 10)


def detect_image(path, model_path):
    dataset = SingleImage(path)
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=4,
                                  drop_last=False)
    file_list = dataset.image_list
    unet = UNet(3, 2).cuda()
    unet.load_state_dict(torch.load(model_path))
    unet.eval()
    unet = torch.nn.DataParallel(unet).cuda()
    # softmax = torch.nn.Softmax(dim=1).cuda()
    num = 0
    for imgs in dataloader:
        imgs = imgs.cuda()
        masks = unet(imgs)
        mask_pred = masks.max(1)[1]

        # masks = softmax(masks)
        mask_pred = mask_pred.detach().cpu().numpy()
        for mask in mask_pred:
            file_path = file_list[num]
            mask = mask.astype(np.uint8)
            out = Image.fromarray(mask)
            out.save(file_path[:-4] + ".png")

            # cv2.imwrite(file_path[:-4] + ".tif",
            #             mask.astype(np.uint8).transpose((1, 2, 0)))
            num += 1
        #     # num += 1


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_path = "05281002_0.6812/05281002_0.6812_netUNet.pth"
    # dir_root = "results/09260133_0.8860"
    image_dirs = "C:/Users/admin/Desktop/paper_cell/Monuseg/eval_dir/img2"
    detect_image(image_dirs, model_path)
