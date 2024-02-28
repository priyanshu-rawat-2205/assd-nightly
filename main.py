import sys
from typing import Optional
from math import dist
import numpy as np
import cv2
import socket
import pickle
from Score import Score
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow, QColorDialog
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtMultimedia import QMediaDevices
from ui.compiled.uiASSD_DASHBOARD import Ui_MainWindow
import threading
import Color
from MultiThreadedHttpServer import RequestHandler, Server
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import requests, json, base64, time
from entity import Entity

class ThreadClass(QThread):
    frameUpdate = Signal(np.ndarray)
    global camPort

    def run(self):
        capture = cv2.VideoCapture(camPort)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = capture.read()
            # flipFrame = cv2.flip(src=frame, flipCode=-1)
            if ret:
                self.frameUpdate.emit(frame)

    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QMainWindow, Ui_MainWindow):
    HEADERSIZE = 10
    averageScore = 0
    count = 0
    checker = False
    center_coordinates = 0
    score = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.ports, self.cams = self.camera_devices()
        self.camDeviceComboBox.addItems(self.cams)
        self.startButton.clicked.connect(self.startWebCam)
        self.stopButton.clicked.connect(self.stopWebcam)
        self.startServerBtn.clicked.connect(self.startServer)
        self.stopServerBtn.clicked.connect(self.stopServer)
        self.clientConnectBtn.clicked.connect(self.startClient)
        self.clientDisconnectBtn.clicked.connect(self.stopClient)
        self.clientGetScoreBtn.clicked.connect(self.getScoreInit)
        self.clientGetScoreBtn.setEnabled(False)
        self.colorComboBox.addItems(Color.colors)


    @Slot(np.ndarray)
    def opencv_emit(self, Image):
        original = self.cvt_cv_qt(Image)
        self.copyImage = Image.copy()
        self.cleanImage = Image.copy()
        cleanTarget = cv2.imread("images/cleanTargetMarked.jpg")
        
        self.camSource.setPixmap(original)
        self.camSource.setScaledContents(True)

        if self.getMaskCheckBox.isChecked():
            # mask_lower_hsv = np.array([self.MaskMinHSlider.value(), self.MaskMinSSlider.value(), self.MaskMinVSlider.value()], dtype=np.uint8)
            # mask_upper_hsv = np.array([self.MaskMaxHSlider.value(), self.MaskMaxSSlider.value(), self.MaskMaxVSlider.value()], dtype=np.uint8)

            hsv_original = cv2.cvtColor(src=self.copyImage,code=cv2.COLOR_BGR2HSV)

            #hardcoded values for masking red color

            mask1_lower_hsv, mask1_upper_hsv, mask2_lower_hsv, mask2_upper_hsv = Color.color(self.colorComboBox.currentText())

            # mask1_lower_hsv = np.array([0,50,50], dtype=np.uint8)
            # mask1_upper_hsv = np.array([10,255,255], dtype=np.uint8)
            mask1 = cv2.inRange(src=hsv_original, lowerb=mask1_lower_hsv, upperb=mask1_upper_hsv)

            # mask2_lower_hsv = np.array([170,50,50], dtype=np.uint8)
            # mask2_upper_hsv = np.array([180,255,255], dtype=np.uint8)
            mask2 = cv2.inRange(src=hsv_original, lowerb=mask2_lower_hsv, upperb=mask2_upper_hsv)

            mask_full = mask1+mask2
            # self.wrapped_mask[np.where(mask=0)] = 0

            # self.mask = cv2.inRange(src=hsv_original, lowerb=mask_lower_hsv, upperb=mask_upper_hsv)
            mask_hsv = hsv_original.copy()
            mask_hsv[np.where(mask_full==0)] = 0

            #conversion of mask from hsv to grayscale
	
            (h,s,v) = cv2.split(mask_hsv)
            v[:] = 100
            img = cv2.merge((v, v, s))
            rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)	
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            self.mask = gray
            # self.mask[np.where(mask_full==0)] = 0

            if self.invertMaskCheckBox.isChecked():
                self.mask = cv2.bitwise_not(self.mask)

            maskQt = self.cvt_cv_qt(self.mask)
            self.maskSource.setPixmap(maskQt)
            self.maskSource.setScaledContents(True)

        if self.detectTargetCheckBox.isChecked():
            contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                points = []
                default = [[0,0], [512,0], [512,512], [0,512]]
                dst = np.array(default, dtype="float32")
                area = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
                if area > self.minAreaSlider.value() and len(approx) == 4:
                    cv2.drawContours(self.copyImage, [approx], 0, (0, 255, 0), 2)
                    n = approx.ravel()
                    i = 0
                    for j in n:
                        if(i % 2 == 0):
                            x = n[i]
                            y = n[i+1]
                            points.append([x,y])
                            string = f"({str(x)},{str(y)})"
                            cv2.putText(self.copyImage, string, (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                        i += 1
                    cntImage = self.cvt_cv_qt(self.copyImage)
                    self.camSource.setPixmap(cntImage)
                    self.camSource.setScaledContents(True)

                    if self.getTargetCheckBox.isChecked():
                        points = np.array(points, dtype="float32")
                        points = self.order_points(points)
                        m = cv2.getPerspectiveTransform(points, dst)
                        wrapped = cv2.warpPerspective(self.cleanImage, m, (512, 512))
                        wrappedImage = self.cvt_cv_qt(wrapped)
                        self.targetSource.setPixmap(wrappedImage)
                        self.targetSource.setScaledContents(True)
                        

                        wrapped_mask_lower_hsv = np.array([self.TargetMinHSlider.value(), self.TargetMinSSlider.value(), self.TargetMinVSlider.value()], dtype=np.uint8)
                        wrapped_mask_upper_hsv = np.array([self.TargetMaxHSlider.value(), self.TargetMaxSSlider.value(), self.TargetMaxVSlider.value()], dtype=np.uint8)

                        hsv_wrapped = cv2.cvtColor(src=wrapped,code=cv2.COLOR_BGR2HSV)
                        self.wrapped_mask = cv2.inRange(src=hsv_wrapped, lowerb=wrapped_mask_lower_hsv, upperb=wrapped_mask_upper_hsv)

                        wrapped_maskQt = self.cvt_cv_qt(self.wrapped_mask)
                        self.targetMaskSource.setPixmap(wrapped_maskQt)
                        self.targetMaskSource.setScaledContents(True)

                        if self.detectShotsCheckBox.isChecked():
                            w_contours, _ = cv2.findContours(self.wrapped_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                            for w_cnt in w_contours:
                                (x2,y2), radius = cv2.minEnclosingCircle(w_cnt)
                                center = (int(x2), int(y2))
                                radius = int(radius)
                                if radius > 5 and radius < 50:
                                    plotTarget = cleanTarget.copy()
                                    cv2.circle(wrapped, center, radius, (255, 0, 0), 2)
                                    cv2.circle(plotTarget, center, 5, (0,0,255), 2)

                                    flipped = cv2.flip(src=plotTarget, flipCode = 1)
                                    flippedQt = self.cvt_cv_qt(flipped)
                                    self.scoreSource.setPixmap(flippedQt)
                                    self.scoreSource.setScaledContents(True)

                                    # cv2.circle(wrapped, (256, 256), 5, (255, 0, 0), 2)
                                    distance = dist(center, (256, 256))
                                    cv2.putText(wrapped, str(distance), center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
                                    wrappedQt = self.cvt_cv_qt(wrapped)
                                    self.targetSource.setPixmap(wrappedQt)
                                    self.targetSource.setScaledContents(True)
                                    score = Score()
                                    self.score = score.finalScore(distance)

                                    # if self.calculateScoreFlag.isChecked():
                                    if Entity.flag:
                                        self.calculateScoreFlag.setChecked(True)
                                        self.averageScore = self.score + self.averageScore
                                        # print(self.averageScore)
                                        self.count += 1
                                        if self.count == 30:
                                            self.finalAverageScore = self.averageScore/30
                                            self.textEdit.append(f"score: {round(self.finalAverageScore, 2)}")
                                            self.textEdit.append("sending...")
                                            # self.broadcast(self.finalAverageScore)
                                            # self.broadcast(center)
                                            Entity.center_coordinates = center
                                            Entity.score = self.finalAverageScore
                                            self.textEdit.append(f"sent")
                                            self.averageScore = 0
                                            self.count = 0
                                            self.calculateScoreFlag.setChecked(False)
                                            Entity.flag=False


                                    self.scoreValueLabel.setText(str(self.score))



    def cvt_cv_qt(self, Image):
        rgb_img = cv2.cvtColor(src=Image,code=cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_img.shape
        bytes_per_line = ch * w
        cvt2QtFormat = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(cvt2QtFormat)

        return pixmap
    
    def startWebCam(self,pin):
        try:
            self.textEdit.append(f"started ({self.camDeviceComboBox.currentText()})")
            self.stopButton.setEnabled(True)
            self.startButton.setEnabled(False)
            self.index = self.cams.index(self.camDeviceComboBox.currentText())

            global camPort
            camPort = self.ports[self.index]
        
        # Opencv QThread
            self.Worker1_Opencv = ThreadClass()
            self.Worker1_Opencv.frameUpdate.connect(self.opencv_emit)
            self.Worker1_Opencv.start()
        

        except Exception as error :
            pass
    
    def stopWebcam(self,pin):
        self.textEdit.append(f"stopped ({self.camDeviceComboBox.currentText()})")
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

        self.Worker1_Opencv.stop()

    def camera_devices(self):

        cams = QMediaDevices.videoInputs()
        ports = []
        devices = []

        for device in cams:
            id = str(device.id())
            id = ''.join(letter for letter in id if letter.isdigit())
            ports.append(int(id))
            devices.append(device.description())

        return ports, devices


    def order_points(self, points):
        rect = np.zeros((4,2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect
    
    def startServer(self, event):

        if not self.serverHost.text() or not self.serverPort.text():
            self.textEdit.append("enter valid address")
            return

        self.host = self.serverHost.text()
        self.port = int(self.serverPort.text())
        self.clients= []

        # Starting Server
        try:
            self.textEdit.append(f"server starting...")

            self.server = Server((self.host, self.port), RequestHandler)
            
            t = threading.Thread(target=self.server.serve_forever)
            t.daemon = True
            t.start()

            # t2 = threading.Thread(target=self.checkFlag)
            # t2.daemon = True
            # t2.start()
            
            self.textEdit.append(f"server is listening on {self.host}:{self.port}")
        except Exception as e:
            self.textEdit.append(f"{type(e)} {e}")


    def broadcast(self, data):
        # response = requests.get(self.url + "score")
        
        self.textEdit.append("score sent")

        
    def stopServer(self, event):
        # self.server.server_close()
        # self.textEdit.append("server closed")
        pass
    
        
    def startClient(self, event):
        host = self.clientHost.text()
        port = self.clientPort.text()
        self.textEdit.append(f"trying to connect...")
        try:
            self.url = "http://" + host + ":" + port + "/"
            response = requests.get(self.url)
            response = json.loads(response.content)
            if response['status'] == "True":
                self.textEdit.append("connected")
                self.clientGetScoreBtn.setEnabled(True)
        except Exception as e:
            self.textEdit.append(f"{type(e)} {e}")
            print(f"{type(e)} {e}")
        


    def stopClient(self, event):
        self.clientSocket.close()
        self.textEdit.append("disconnected from server")

    def getScoreInit(self):
        self.textEdit.append("refreshing score...")
        self.clientGetScoreBtn.setEnabled(False)

        response = requests.get(self.url + "prepare")
        response = json.loads(response.content)
        if response['status'] == "True":
            print('score prepared')
            time.sleep(5)
            score = requests.get(self.url + "score")
            score = json.loads(score.content)
            self.textEdit.append(str(score['score']))
            print(score['score'])
            self.clientGetScoreBtn.setEnabled(True)
        
            

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())