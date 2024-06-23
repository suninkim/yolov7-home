from __future__ import print_function

import logging
import os
import sys
from concurrent import futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import cv2
import grpc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from models.experimental import attempt_load
from numpy import random

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, select_device, time_synchronized


class DetectionNode:
    def __init__(self, cfg):

        # model definition
        weights = cfg["model"]["weights"]
        imgsz = cfg["model"]["img_size"]
        trace = cfg["model"]["no_trace"]
        self.augment = cfg["model"]["augment"]

        # Initialize
        set_logging()
        self.device = select_device(cfg["model"]["device"])
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Parameter
        self.conf_thres = cfg["model"]["conf_thres"]
        self.iou_thres = cfg["model"]["iou_thres"]
        self.classes = cfg["model"]["classes"]
        self.agnostic_nms = cfg["model"]["agnostic_nms"]

        # Load model
        print(f"weights: {weights}")
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(imgsz, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            self.model.half()  # to FP16
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.img_size, self.img_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

    def detect(self, image):

        # Padded resize
        img = letterbox(image, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
        )
        t3 = time_synchronized()

        result = {}

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], image.shape
                ).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    result[f"{self.names[int(c)]}"] = n

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f"{self.names[int(cls)]} {conf:.2f}"
                    plot_one_box(
                        xyxy,
                        image,
                        label=label,
                        color=self.colors[int(cls)],
                        line_thickness=1,
                    )

            # Print time (inference + NMS)
            print(
                f"Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
            )

        return result, image


if __name__ == "__main__":
    logging.basicConfig()
