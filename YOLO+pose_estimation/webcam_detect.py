import argparse
import chainer
import sys
import cv2
import torch
from numpy import random
import re
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.pose_detector import PoseDetector, draw_person_pose
from utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging
from utils.torch_utils import select_device
from multiprocessing import Process, Queue
import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QStringListModel
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap

form_window = uic.loadUiType('./utils/exercise.ui')[0]

# producer
def setup(q1, q2, q8, weights='yolov5s.pt', imgsz=640, iou=0.5, classes=None,
          agnostic=False, augment=False):

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    conf = 0.3
    while True:
        if not q1.empty():
            if not q8.empty():
                conf = q8.get()
            frame = q1.get()
            yolo(frame, q2, imgsz, device, half, model, augment, conf, iou, classes, agnostic, names, colors)


def yolo(q1, q2, imgsz, device, half, model, augment, conf, iou, classes, agnostic_nms, names, colors):
    source = q1
    dataset = LoadImages(source, img_size=imgsz)
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf, iou, classes=classes, agnostic=agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if len(det) >= 2:
                    det_num = []
                    for i in det:
                        det_num.append((i[2] + i[3]) - (i[0] + i[1]))
                    largest = det_num[0]
                    for i in range(len(det_num)):
                        if det_num[i] > largest:
                            largest = det_num[i]
                    det = [det[det_num.index(largest)]]

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    conf = re.findall("tensor\((.*).", str(conf).split(", ")[0])[0]
                    conf = round(float(conf) * 100, 1)
                    label = f"{names[int(cls)]} {conf}%"
                    num = []
                    f = str(xyxy).split(", ")
                    for string in f:
                        axes = re.findall("tensor\((.*).", string)
                        if axes != []:
                            num.append(int(axes[0]))
                    outq = num, label, colors[int(cls)], names[int(cls)]
                    q2.put(outq)
        break

def pose_estimation(q1, q3, q4):
    chainer.config.enable_backprop = False
    chainer.config.train = False
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=0)
    num = [0, 0, 640, 480]
    j = 0
    while True:
        if not q4.empty():
            before_num = num
            num = q4.get()
            if before_num[0] != 0 and j % 15 != 0:
                if before_num[0] < num[0]:
                    num[0] = before_num[0]
                if before_num[1] < num[1]:
                    num[1] = before_num[1]
                if before_num[2] > num[2]:
                    num[2] = before_num[2]
                if before_num[3] > num[3]:
                    num[3] = before_num[3]
            else:
                pass
            j+=1
        if not q1.empty():
            try:
                img = q1.get()[num[1]:num[3], num[0]:num[2]]
                poses, _ = pose_detector(img)
                outq3 = poses, num
                q3.put(outq3)
            except TypeError:
                pass

def webcam_producer(q1, q7, source=0):
    capture = cv2.VideoCapture(source)
    if source == 0:
        while True:
            ret, frame = capture.read()
            q1.put(frame)
        capture.release()
    else:
        fps = capture.get(cv2.CAP_PROP_FPS)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = capture.read()
            if ret:
                cv2.waitKey(33)
                q1.put(frame)
            else:
                break
            outq7 = fps, w, h, source, ret
            q7.put(outq7)
        capture.release()

def webcam_out(q1, q2, q3, q4, q5, q6, q7):
    Squat_score, Bench_score, Dead_score  = 0, 0, 0
    Squat_Before_flag, Bench_Before_flag, Dead_Before_flag = False, False, False
    fourcc = 'mp4v'  # output video codec
    x = 0
    while True:
        if not q1.empty():
            frame = q1.get()
            if not q2.empty():
                q2num, label, colors, name = q2.get()
                q4.put(q2num)
            if not q3.empty():
                poses, num = q3.get()
            try:
                frame2 = frame[num[1]:num[3], num[0]:num[2]].copy()
                canvas, Squat_score, Bench_score, Dead_score, Squat_Before_flag, Bench_Before_flag, Dead_Before_flag, \
                squat_status, bench_status, dead_status = draw_person_pose(frame2, poses, Squat_score, Bench_score, Dead_score,\
                     Squat_Before_flag, Bench_Before_flag, Dead_Before_flag, name)
                frame[num[1]:num[3], num[0]:num[2]] = canvas
                outq6 = Squat_score, Bench_score, Dead_score
                q6.put(outq6)
            except:
                pass
            try:
                plot_one_box(q2num, frame, label=label, color=colors, line_thickness=3)
            except:
                pass
            q5.put(frame)
        if not q7.empty():
            fps, w, h, source, ret = q7.get()
            try:
                vid_writer
            except UnboundLocalError:
                vid_writer = cv2.VideoWriter(f"{source[:-4]}_pose_estimation.mp4", cv2.VideoWriter_fourcc(*fourcc), fps,
                                             (w, h))
            try:
                vid_writer.write(frame)
            except UnboundLocalError:
                pass
            x = 0
        elif q7.empty():
            try:
                vid_writer
                if x == 2000:
                    vid_writer.release()
                    break
            except UnboundLocalError:
                pass
            x += 1

class consumer1(QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self, q5):
        super().__init__()
        self.q5 = q5

    def run(self):
        while True:
            if not self.q5.empty():
                rgbImage = cv2.cvtColor(q5.get(), cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                data1 = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(data1)

class consumer2(QThread):
    renewal = pyqtSignal(tuple)
    def __init__(self, q6):
        super().__init__()
        self.q6 = q6

    def run(self):
        while True:
            if not self.q6.empty():
                data2 = q6.get()
                self.renewal.emit(data2)

class Exam(QWidget, form_window):
    def __init__(self, q1, q2, q3, q4, q5, q6, q7, q8):
        super().__init__()
        self.setupUi(self)
        self.init_UI()
        self.consumer1 = consumer1(q5)
        self.consumer1.changePixmap.connect(self.setImage)
        self.consumer1.start()
        self.consumer2 = consumer2(q6)
        self.consumer2.renewal.connect(self.label_change)
        self.consumer2.start()
        self.btn_webcam.clicked.connect(self.webcam)
        self.btn_upload.clicked.connect(self.upload)
        self.spn_conf.valueChanged.connect(self.conf_change)
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.q6 = q6
        self.q7 = q7
        self.q8 = q8

    def init_UI(self):
        self.p1_flag = False

    @pyqtSlot(QImage)
    def setImage(self, data1):
        self.lbl_image.setPixmap(QPixmap.fromImage(data1))

    @pyqtSlot(tuple)
    def label_change(self, data1):
        Squat_score, Bench_score, Dead_score = data1
        self.lbl_squat.setText(str(Squat_score))
        self.lbl_bench.setText(str(Bench_score))
        self.lbl_dead.setText(str(Dead_score))

    def webcam(self):
        if self.p1_flag == True:
            self.p1.kill()
            self.p4.kill()
        self.p1 = Process(name="webcam_producer", target=webcam_producer, args=(self.q1, self.q7, ), daemon=True)
        self.p1.start()
        self.p4 = Process(name="webcam_out", target=webcam_out, args=(self.q1, self.q2, self.q3, self.q4, self.q5,\
                                                                      self.q6, self.q7, ), daemon=True)
        self.p4.start()
        self.p1_flag = True

    def upload(self):
        if self.p1_flag == True:
            self.p1.kill()
            self.p4.kill()
        self.path = QFileDialog.getOpenFileName(self, "Open file", '', "mp4 file(*.mp4);;avi files(*.avi);;All files(*.*)", '')
        if self.path[0]:
            self.p1 = Process(name="webcam_producer", target=webcam_producer, args=(self.q1, self.q7, self.path[0], ), daemon=True)
            self.p1.start()
            self.p4 = Process(name="webcam_out", target=webcam_out, args=(self.q1, self.q2, self.q3, self.q4, self.q5, \
                                                                          self.q6, self.q7,), daemon=True)
            self.p4.start()
            self.p1_flag = True

    def conf_change(self):
        self.conf = (self.lcd_conf.intValue()/100)
        if self.p1_flag == True:
            self.q8.put(self.conf)

if __name__ == "__main__":
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    q5 = Queue()
    q6 = Queue()
    q7 = Queue()
    q8 = Queue()

    p2 = Process(name="yolo", target=setup, args=(q1, q2, q8, 'weights/exercise3_m_best.pt', ), daemon=True)
    p3 = Process(name='pose_estimation', target=pose_estimation, args=(q1, q3, q4, ), daemon=True)


    p2.start()
    p3.start()

    # Main process
    app = QApplication(sys.argv)
    MainWindow = Exam(q1, q2, q3, q4, q5, q6, q7, q8)
    MainWindow.show()
    sys.exit(app.exec_())