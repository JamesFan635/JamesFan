import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import onnxruntime
# 確保這些函數能夠處理numpy數組
from utils.datasets import LoadImages
from utils.general import scale_coords, xyxy2xywh
from utils.plots import plot_one_box


def detect(save_img=False):
    source, weights, imgsz, conf_thres, iou_thres = opt.source, opt.weights, opt.img_size, opt.conf_thres, opt.iou_thres
    save_img =  source.endswith('.jpg')  # save inference images

    # Initialize
    device = 'cpu'
    half = False  # ONNX Runtime 不支持半精度浮点

    # Load ONNX model
    session = onnxruntime.InferenceSession(weights)

    if source == '0':
        source = 0  # opencv使用0來存取攝像頭
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = ['class1', 'class2', 'class3']  # 示例类名，根据实际情况替换
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]


    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # 確保圖像為模型預期的維度 (3, 640, 640)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # 調整通道順序為 C x H x W
        img = np.expand_dims(img, axis=0)

        # 確保輸入維度正確
        if img.shape != (1, 3, imgsz, imgsz):
            raise ValueError(f'圖像維度不正確，模型期望的輸入維度為 (1, 3, {imgsz}, {imgsz})，但獲得 {img.shape}')

        # Inference
        t1 = time.time()
        inputs = {session.get_inputs()[0].name: img}
        pred = session.run(None, inputs)[0]



        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            print(f'{s}Done. ({(time.time() - t1):.3f}s)')

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(str(Path(opt.save_dir) / Path(p).name), im0)

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',  type=str, default='yolov7.onnx', help='model.onnx path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()


    detect()