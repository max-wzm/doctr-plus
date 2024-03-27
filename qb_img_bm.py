import os
from os.path import join as pjoin

import cv2
from infer import request_infer_with_bm
import hdf5storage as h5


class QbBmProcessor:
    def __init__(self, dataroot="./data/QBDoc") -> None:
        self.dataroot = dataroot
        self.all_samples = [
            x[:-4]
            for x in os.listdir(pjoin(self.dataroot, "img"))
            if x.endswith(".jpg")
        ]

    def process(self):
        for sample_id in self.all_samples:
            img_path = pjoin(self.dataroot, "img", f"{sample_id}.jpg")
            bm_path = pjoin(self.dataroot, "bm", f"{sample_id}.mat")
            img_RGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:, :, ::-1]
            if os.path.exists(bm_path):
                continue
            bm = request_infer_with_bm(img_RGB)
            h5.savemat(bm_path, {"bm": bm})


q = QbBmProcessor()
q.process()
