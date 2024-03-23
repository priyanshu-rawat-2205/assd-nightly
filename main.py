import sys
from math import dist, floor
import numpy as np
import cv2
import socket
from Score import Score
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaDevices
from ui.compiled.uiASSD_DASHBOARD import Ui_MainWindow
import threading
import Color
import json, time
from _thread import *
from datetime import datetime

import os

class ThreadClass(QThread):
    frameUpdate = Signal(np.ndarray)
    global camPort
    url = "http://192.168.118.214:8080/video"
    def run(self):
        capture = cv2.VideoCapture(camPort)
        # capture = cv2.VideoCapture(self.url)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = capture.read()
            # flipFrame = cv2.flip(src=frame, flipCode=-1)
            flipFrame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if ret:
                # self.frameUpdate.emit(frame)
                self.frameUpdate.emit(flipFrame)


    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QMainWindow, Ui_MainWindow):
    HEADERSIZE = 10
    averageScore = 0
    count = 0
    checker = False
    lock = threading.Lock()
    clientCopy = None
    SCALE_CONTENTS = True

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

        lab = cv2.cvtColor(Image, cv2.COLOR_BGR2Lab)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        # lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # original = self.cvt_cv_qt(Image)
        original = self.cvt_cv_qt(bgr)


        # self.copyImage = Image.copy()
        self.copyImage = bgr.copy()
        self.cleanImage = Image.copy()
        cleanTarget = cv2.imread(self.resource_path("images/cleanTargetCropped1.jpg"))
        # cleanTarget = cv2.imread("images/cleanTargetCropped1.jpg")

        
        self.camSource.setPixmap(original)
        self.camSource.setScaledContents(self.SCALE_CONTENTS)

        if self.getMaskCheckBox.isChecked():
            
            hsv_original = cv2.cvtColor(src=self.copyImage,code=cv2.COLOR_BGR2HSV)

            #hardcoded values for masking red color

            mask1_lower_hsv, mask1_upper_hsv, mask2_lower_hsv, mask2_upper_hsv = Color.color(self.colorComboBox.currentText())

            mask1 = cv2.inRange(src=hsv_original, lowerb=mask1_lower_hsv, upperb=mask1_upper_hsv)

            mask2 = cv2.inRange(src=hsv_original, lowerb=mask2_lower_hsv, upperb=mask2_upper_hsv)

            mask_full = mask1+mask2

            mask_hsv = hsv_original.copy()
            mask_hsv[np.where(mask_full==0)] = 0

            #conversion of mask from hsv to grayscale
	
            # (h,s,v) = cv2.split(mask_hsv)
            # v[:] = 100
            # img = cv2.merge((v, v, s))
            # rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)	
            # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            mask_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_HSV2BGR)
            mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

            #correction of non-uniform illumination and various camera noises
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # morph = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
            # out_gray=cv2.divide(image, bg, scale=255)
            # out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]


            # self.mask = mask_gray
            self.mask = mask_gray
            
            if self.invertMaskCheckBox.isChecked():
                self.mask = cv2.bitwise_not(self.mask)

            maskQt = self.cvt_cv_qt(self.mask)
            self.maskSource.setPixmap(maskQt)
            self.maskSource.setScaledContents(self.SCALE_CONTENTS)

        if self.detectTargetCheckBox.isChecked():
            contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                points = []
                default = [[0,0], [512,0], [512,512], [0,512]]
                dst = np.array(default, dtype="float32")
                area = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
                # if area > self.minAreaSlider.value() and len(approx) == 4:
                if area > 500 and len(approx) == 4:
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
                    self.camSource.setScaledContents(self.SCALE_CONTENTS)

                    if self.getTargetCheckBox.isChecked():
                        points = np.array(points, dtype="float32")
                        points = self.order_points(points)
                        m = cv2.getPerspectiveTransform(points, dst)
                        wrapped = cv2.warpPerspective(self.cleanImage, m, (512, 512))
                        # wrapped = cv2.rotate(wrapped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        # wrapped = cv2.flip(wrapped, 0)
                        wrappedImage = self.cvt_cv_qt(wrapped)
                        self.targetSource.setPixmap(wrappedImage)
                        self.targetSource.setScaledContents(self.SCALE_CONTENTS)
                        

                        # wrapped_mask_lower_hsv = np.array([self.TargetMinHSlider.value(), self.TargetMinSSlider.value(), self.TargetMinVSlider.value()], dtype=np.uint8)
                        # wrapped_mask_upper_hsv = np.array([self.TargetMaxHSlider.value(), self.TargetMaxSSlider.value(), self.TargetMaxVSlider.value()], dtype=np.uint8)

                        hsv_wrapped = cv2.cvtColor(src=wrapped,code=cv2.COLOR_BGR2HSV)
                        # self.wrapped_mask = cv2.inRange(src=hsv_wrapped, lowerb=wrapped_mask_lower_hsv, upperb=wrapped_mask_upper_hsv)

                        #hardcode value
                        wrapped_mask1 = cv2.inRange(hsv_wrapped, lowerb=mask1_lower_hsv, upperb=mask1_upper_hsv)
                        wrapped_mask2 = cv2.inRange(hsv_wrapped, lowerb=mask2_lower_hsv, upperb=mask2_upper_hsv)

                        wrapped_mask_full = wrapped_mask1 + wrapped_mask2
                        wrapped_mask_hsv = hsv_wrapped.copy()
                        wrapped_mask_hsv[np.where(wrapped_mask_full==0)] = 0

                        wrapped_mask_bgr = cv2.cvtColor(wrapped_mask_hsv, cv2.COLOR_HSV2BGR)
                        wrapped_mask_gray = cv2.cvtColor(wrapped_mask_bgr, cv2.COLOR_BGR2GRAY)



                        wrapped_maskQt = self.cvt_cv_qt(wrapped_mask_gray)
                        self.targetMaskSource.setPixmap(wrapped_maskQt)
                        self.targetMaskSource.setScaledContents(self.SCALE_CONTENTS)

                        if self.detectShotsCheckBox.isChecked():
                            # w_contours, _ = cv2.findContours(self.wrapped_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            w_contours, _ = cv2.findContours(wrapped_mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            
                            # if len(w_contours) == 1: # added this countour check might have to remove later
                            # print(len(w_contours))
                            if len(w_contours) != 1: # added this countour check might have to remove later


                                for w_cnt in w_contours:
                                    (x2,y2), radius = cv2.minEnclosingCircle(w_cnt)
                                    center = (int(x2), int(y2))
                                    radius = int(radius)
                                    if radius > 15 and radius < 50:

                                        score = Score()

                                        (x_cor, y_cor) = center
                                        (x_cor, y_cor) = (score.px2cm(x_cor), score.px2cm(y_cor))
                                        (x_cor, y_cor) = (floor(score.cm2px(x_cor)), floor(score.cm2px(y_cor)))


                                        plotTarget = cleanTarget.copy()
                                        cv2.circle(wrapped, center, radius, (255, 0, 0), 2)
                                        # cv2.circle(plotTarget, (x_cor, y_cor), 5, (0,0,255), 2)
                                        # self.textEdit.append(f"{x_cor}, {y_cor}")
                                        # (a, b) = center
                                        cv2.circle(plotTarget, center, 5, (0,0,255), 2)
                                        # self.textEdit.append(f"{center}")


                                        # flipped = cv2.flip(src=plotTarget, flipCode = 1)
                                        # flippedQt = self.cvt_cv_qt(flipped)
                                        flippedQt = self.cvt_cv_qt(plotTarget) #added the non-flipped image for testing might remove later

                                        self.scoreSource.setPixmap(flippedQt)
                                        self.scoreSource.setScaledContents(self.SCALE_CONTENTS)
                                       

                                        # cv2.circle(wrapped, (256, 256), 5, (255, 0, 0), 2)
                                        distance = dist(center, (256, 256))
                                        cv2.putText(wrapped, str(distance), center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
                                        # cv2.putText(wrapped, str(distance), (x_cor, y_cor), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
                                        wrappedQt = self.cvt_cv_qt(wrapped)
                                        self.targetSource.setPixmap(wrappedQt)
                                        self.targetSource.setScaledContents(self.SCALE_CONTENTS)
                                        
                                        self.score = score.finalScore(distance)

                                        # if self.calculateScoreFlag.isChecked():
                                        if self.calculateScoreFlag.isChecked() and self.clientStatus : #added this check to prevent brodcasting data on bad descripter
                                            self.averageScore = self.score + self.averageScore
                                            # print(self.averageScore)
                                            # self.textEdit.append("calculating score...")
                                            self.count += 1
                                            if self.count == 30:
                                                self.finalAverageScore = self.averageScore/30
                                                self.textEdit.append(f"score: {round(self.finalAverageScore, 2)}")
                                                if(self.finalAverageScore >= 7.0):
                                                    if self.ManualModeRadioButton.isChecked():
                                                        self.broadcast({'score':self.finalAverageScore, 'center': center, 'mode': 'manual'})
                                                        self.calculateScoreFlag.setChecked(False)
                                                    else:
                                                        self.broadcast({'score':self.finalAverageScore, 'center': center, 'mode': 'auto'})
                                                        time.sleep(5)
                                                    self.textEdit.append("sending...")
                                                    # self.broadcast(self.finalAverageScore)S
                                                    # self.broadcast(center)

                                                    
                                                        
                                                    # self.broadcast({'score':self.finalAverageScore, 'center':(x_cor, y_cor)})
                                                    # Entity.center_coordinates = center
                                                    # Entity.score = self.finalAverageScore
                                                    self.textEdit.append(f"sent")
                                                    self.averageScore = 0
                                                    self.count = 0
                                                    

                                                    # Entity.flag=False


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
        self.clients= [] #this list might be the cause of seg faults as it might raise bad file discriptor error

        # Starting Server
        try:
            self.textEdit.append(f"server starting...")
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind((self.host, self.port))

            server.listen()
            self.server = server

            # start_new_thread(self.acceptClients, (server, ))
            t1 = threading.Thread(target = self.acceptClients, args = (server, ))
            t1.start()
            self.textEdit.append(f"server is listening on {self.host}:{self.port}")
        except Exception as e:
            self.textEdit.append(f"{type(e)} {e}")

    
    def acceptClients(self, server):
        while True:
            try:
                # establish connection with client
                print('accepting clients')
                client, addr = server.accept()
                self.clients.append(client)
        
                # lock acquired by client
                self.lock.acquire()
                print('Connected to :', addr[0], ':', addr[1])
                self.textEdit.append(f"connected to: {addr[0]} : {addr[1]}")
                self.clientStatus = True
                if self.AutoModeRadioButton.isChecked():
                    self.calculateScoreFlag.setChecked(True)
        
                # Start a new thread and return its identifier
                # start_new_thread(self.requestHandler, (client,))
                t1 = threading.Thread(target = self.requestHandler, args = (client, ))
                t1.start()
                t1.join()
                client.close()
                print("came out of request handler thread")
            except Exception as e:
                server.close()
                self.clients = []
                self.textEdit.append(f"{type(e)} {e}")
                print(f"{type(e)} {e}")
                return

    def requestHandler(self, client):
        while True:
            try: 
                # data received from client
                data = client.recv(1024)
                if not data:
                    print('Bye')
                    self.textEdit.append("client disconnected")
                    
                    # lock released on exit
                    self.lock.release()
                    client.close()
                    self.clientStatus = False
                    return
                    # break
                data = json.loads(data)
                if(data['status'] == 'True'):
                    self.calculateScoreFlag.setChecked(True)
                    self.textEdit.append("score requested")
                elif(data['status'] == 'False'):
                    self.textEdit.append("client disconnected")
                    print("client disconnected")
                    self.clientStatus = False
                    self.lock.release()
                    client.close()
                    return
                    # break
                    
                # reverse the given string from client
                # data = data[::-1]
        
                # send back reversed string to client
                # client.send(data)

            except Exception as e:
                print(e)
                client.close()
                sys.exit()
                # break


    def broadcast(self, data):
        try:                
            data = json.dumps(data)
            client = self.clients[0]
            client.send(bytes(data, "utf-8"))      
            self.textEdit.append("score sent")
        except Exception as e:
            print(f"brodcast error: {type(e)}: {e}")
            self.textEdit.append('failed to send score')

        
    def stopServer(self, event):
        try:
            self.server.close()
            self.textEdit.append("server closed")
        except Exception as e:
            print(e)
            self.textEdit.append(e)
    
        
    def startClient(self, event):
        host = self.clientHost.text()
        port = int(self.clientPort.text())
        self.textEdit.append(f"trying to connect...")
        try:
            # self.url = "http://" + host + ":" + port + "/"
            # response = requests.get(self.url)
            # response = json.loads(response.content)
            # if response['status'] == "True":
            #     self.textEdit.append("connected")
            #     self.clientGetScoreBtn.setEnabled(True)
            client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            client.connect((host, port))
            self.textEdit.append("connected")
            self.clientGetScoreBtn.setEnabled(True)
            self.clientCopy = client
            start_new_thread(self.clientHandler, (client, ))

        except Exception as e:
            self.textEdit.append(f"{type(e)} {e}")
            print(f"{type(e)} {e}")
    
    def clientHandler(self, client):
        # message = 'test'
        while True:
            try:
                data = client.recv(1024)
                data = json.loads(data)
                
                if (type(data) == type({'type':'dict'})):
                        self.clientGetScoreBtn.setEnabled(False)
                        self.scoreValueLabel.setText(str(round(data['score'], 1)))
                        cleanTarget = cv2.imread(self.resource_path("images/cleanTargetCropped1.jpg"))
                        # cleanTarget = cv2.imread("images/cleanTargetCropped1.jpg")
                        plot = cv2.circle(cleanTarget, data['center'], 5, (0,0,255), 2)

                        #write shot target images to path
                        self.save_image(plot)

                        # plot = cv2.flip(src=plot, flipCode=1)
                        plotQt = self.cvt_cv_qt(plot)
                        self.scoreSource.setPixmap(plotQt)
                        self.scoreSource.setScaledContents(True)
                        if data['mode'] == 'auto':
                            self.clientGetScoreBtn.setEnabled(False)
                        else:
                            self.clientGetScoreBtn.setEnabled(True)
                        self.textEdit.append("score refreshed")
                else:
                    self.textEdit.append("failed to fetch score") 

            except:
                client.close()
                break

    def stopClient(self, event):
        request = json.dumps({'status' : 'False'})
        self.clientCopy.send(bytes(request, 'utf-8'))
        self.textEdit.append("disconnected from server")
        self.clientCopy.close()

    def getScoreInit(self):
        self.textEdit.append("refreshing score...")
        self.clientGetScoreBtn.setEnabled(False)

        request = json.dumps({'status' : 'True'})
        self.clientCopy.send(bytes(request, 'utf-8'))
        
    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path) 

    def save_image(self, image):
        base_path = f"/home/{os.environ.get('USER')}/"
        save_dir = ".ASSD/"
        now = datetime.now().strftime("%d-%m-%Y[%H:%M]")
        img_name = datetime.now().strftime("[%H:%M:%S]")
        if not os.path.exists(base_path + save_dir): 
            os.mkdir(os.path.join(base_path, save_dir))
        if not os.path.exists(base_path + save_dir + now):
            os.mkdir(os.path.join(base_path + save_dir, now))
        cv2.imwrite(os.path.join(base_path + save_dir + now, f"shot{img_name}.jpg"), image)   

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())