import argparse
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.core_net import GeoTr
from infer.process import post_process

warnings.filterwarnings("ignore")


class GeoTrP(nn.Module):
    def __init__(self):
        super(GeoTrP, self).__init__()
        self.GeoTr = GeoTr()

    def forward(self, x):
        bm = self.GeoTr(x)  # [0]
        bm = 2 * (bm / 288) - 1

        bm = (bm + 1) / 2 * 448

        bm = F.interpolate(bm, size=(448, 448), mode="bilinear", align_corners=True)

        return bm

    def do_infer(self, img_cv2):
        im_ori = img_cv2 / 255.0
        h_, w_, c_ = im_ori.shape
        raw = im_ori.copy()
        im_ori = cv2.resize(im_ori, (448, 448))

        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            # geometric unwarping
            bm = PROXY(im.cuda())
            print(f"bf shape: {bm.shape}")
            bm = bm.cpu().numpy()[0]
            print(f"af shape: {bm.shape}")
            print(bm)
            # bm0 = bm[0, :, :]
            # bm1 = bm[1, :, :]
            # bm0 = cv2.blur(bm0, (3, 3))
            # bm1 = cv2.blur(bm1, (3, 3))

            # img_geo = cv2.remap(im_ori, bm0, bm1, cv2.INTER_LINEAR)*255
            # img_geo = cv2.resize(img_geo, (w_, h_))
            # out = img_geo.astype(np.uint8)  # save

            bm_flow = bm
            img_geo, *_ = post_process(raw, bm_flow)
            out = img_geo.astype(np.uint8)  # save
        return out


PROXY: GeoTrP = None


def init():
    global PROXY
    PROXY = GeoTrP().cuda()
    # reload geometric unwarping model
    reload_model(PROXY.GeoTr, "./model_save/model.pt")

    # To eval mode
    PROXY.eval()


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location="cuda:0")
        print(len(pretrained_dict.keys()))
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def infer(img_cv):
    """
    receive and return an array
    """
    if not PROXY:
        init()
    return [PROXY.do_infer(img_cv2=img_cv) for _ in range(1)][0]


def rec(opt):
    # print(torch.__version__) # 1.5.1
    img_list = os.listdir(opt.distorrted_path)  # distorted images list

    if not os.path.exists(opt.gsave_path):  # create save path
        os.mkdir(opt.gsave_path)

    for img_path in img_list:
        name = img_path.split(".")[-2]  # image name

        img_path = opt.distorrted_path + img_path  # read image and to tensor
        im_ori = cv2.imread(img_path)
        out = PROXY.do_infer(im_ori)
        cv2.imwrite(opt.gsave_path + name + "_geo" + ".png", out)  # save

        print("Done: ", img_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distorrted_path", default="./distorted/")
    parser.add_argument("--gsave_path", default="./rectified/")
    parser.add_argument("--GeoTr_path", default="./model_save/model.pt")

    opt = parser.parse_args()
    init(opt)
    rec(opt)


if __name__ == "__main__":
    imori = cv2.imread("/data/home/mackswang/doc_rect/My-DocTr-Plus/distorted/pic.jpg")
    out = infer(imori)
    cv2.imwrite("out.png", out)
