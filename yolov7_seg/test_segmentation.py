# from models.common import DetectMultiBackend
# from utils.general import check_img_size, non_max_suppression, scale_coords
# from utils.segment.general import process_mask

from yolov7_seg import DetectMultiBackend
from yolov7_seg import check_img_size, non_max_suppression, scale_coords
from yolov7_seg import process_mask

import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np


class Segmentor:

    def __init__(self, weights, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45,  max_det=1000,
                 masks=32) -> None:
        self.weights_path = weights
        self.input_size=imgsz
        self.confidence_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.max_detections = max_det
        self.device = torch.device('cuda:0')
        self.masks = masks

    def load(self):
        self.net = DetectMultiBackend(self.weights_path, device=self.device, fp16=True)
        self.input_size = check_img_size(self.input_size, s=self.net.stride)
        cudnn.benchmark = True
        self.net.warmup((1,3,*self.input_size))

    def preprocess(self, image):
        # im = letterbox(image, self.input_size, stride=self.net.stride)[0]  # padded resize
        im = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.net.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None] 
        return im

    def detect(self, image):
        image = cv2.imread(image)
        preprocessed_image = self.preprocess(image)
        pred, out = self.net.model(preprocessed_image)
        proto = out[1]
        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold,
                                   max_det=self.max_detections, nm=self.masks)
        self.postprocess(pred,proto, image)
        
    def postprocess(self, detections, proto, image):
        # pre_shape = np.array(preprocessed_image.shape[2:])
        for i, det in enumerate(detections):  # per image
            # annotator = Annotator(image, line_width=3, example=str(self.net.names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], self.input_size, upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.input_size, det[:, :4], image.shape).round()

                # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                    # s = f"ind:{i} det:{n} class:{self.net.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting -----------------------------------------------------------------------------------
                masks = masks.permute(1,2,0).contiguous().byte().cpu().numpy()
                my_mask = masks
                masks = masks.repeat(3,2)
                new_image = image.copy()
                contours = cv2.findContours(cv2.resize(my_mask, (image.shape[1], image.shape[0])), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                new_image = cv2.fillPoly(img=new_image, pts=contours[0], color=(0,255,0))
                new_image = cv2.addWeighted(src1=image, alpha=1 - .5, src2=new_image, beta=.5, gamma=0)
                cv2.namedWindow('mytest2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
                cv2.resizeWindow('mytest2', new_image.shape[1], new_image.shape[0])
                cv2.imshow('mytest2', new_image)
                cv2.waitKey()  # 1 millisecond
                cv2.destroyAllWindows()
                # Mask plotting ----------------------------------------------------------------------------------------

                # Write results
                # for *xyxy, conf, cls in reversed(det[:, :6]):
                #     c = int(cls)  # integer class
                #     label = f'{self.net.names[c]} {conf:.2f}'
                #     annotator.box_label(xyxy, label, color=colors(c, True))


if __name__ == '__main__':
    seg = Segmentor('/home/rodwin/repos/own/yolov7_segmentation/yolov7_seg/best.pt')
    seg.load()
    seg.detect('/home/rodwin/repos/own/yolov7_segmentation/yolov7_seg/mpv-shot0063.jpg')