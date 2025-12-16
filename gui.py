from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QSlider, QWidget, QPushButton, QFileDialog, QSizePolicy, QGridLayout,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QTabWidget, QLabel, QComboBox, QRadioButton, 
                             QButtonGroup, QToolTip, QWidgetAction)
from PyQt6.QtGui import (QCursor, QImage, QMouseEvent, QPixmap)
from typing import cast

from matplotlib.pyplot import winter
import numpy as np
from plotter import Plot
from guico import Img, Points

from collections import deque

import sys
import cv2 as cv

class ClickLable(QLabel):
    clicked = pyqtSignal(int,int)

    def __init__(self):
        super().__init__()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(ev.position().x()),int(ev.position().y()))


class Mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.options = ('original','grayscale', 'BW', 
                        'neg', 'RGB','contrast','convolv',
                        'hsvMask','segmentation','morphology','point')
        
        self.menu = ('file',)
        self.widgetdata ={}    
        
        mainlayout = QTabWidget()
        mainlayout.setMovable(True)
        mainlayout.setTabPosition(QTabWidget.TabPosition.West)
        self.setCentralWidget(mainlayout)

        for op in self.options:
            tab = QWidget()
            mainlayout.addTab(tab,op)

            setattr(self, op, tab)
        self._setup(self.options)

        menulayout = QHBoxLayout()
        menu = QWidget()
        menu.setLayout(menulayout)
        self.setMenuWidget(menu)

        for m in self.menu:
            button = QPushButton(m)
            menulayout.addWidget(button)

            setattr(self,m,button)
        self._setup(self.menu)    

    
    def _setup(self, components) -> None:
        for component in components:
            field, func = getattr(self,component), getattr(self, f'_handle{component}')
            func(field) if field and func else None

    #tab handler        
    def _handleoriginal(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path': ''}

        widget.setStyleSheet('background-color: black')

        vlayout = QVBoxLayout()
        widget.setLayout(vlayout)

        # container untuk gambar dan chart
        container = QWidget()
        Himglayout = QGridLayout()
        container.setLayout(Himglayout)

        # isi container
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        Himglayout.addWidget(label_img,0,0)
        Himglayout.addWidget(label_chart,0,1)
        Himglayout.setSpacing(2)
        Himglayout.setContentsMargins(0,0,0,0)

        # tombol proses
        button_proses = QPushButton('Load Img')
        button_proses.setStyleSheet('background-color: dark gray')

        # tambahkan ke layout utama
        vlayout.addWidget(container, stretch=2, alignment=Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(button_proses, stretch=1, alignment=Qt.AlignmentFlag.AlignBottom)

        def loadonclick():
            if not self.widgetdata[widget]['path']:
                label_img.setText('Load file terlebih dahulu')
                label_img.setStyleSheet('background-color: red')
                return
            
            datadict = self.widgetdata[widget]
            #try to load file
            try:
                datadict['img'] = Img(datadict['path'])
                self._display_to_label(label_img,self._cv2_to_pixmap(datadict['img'].img))
                buf = Plot.makePlot(datadict['img'].img).getBuf()

            # ubah buffer → QPixmap
                qimg = QImage.fromData(buf.read())
                pixmap = QPixmap.fromImage(qimg)
                self._display_to_label(label_chart, pixmap)

            except Exception as e:
                label_img.setText(f'err :{e}')
        
        button_proses.clicked.connect(loadonclick)
    
    def _handlemorphology(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path':'', 'reference': None}
        widget.setStyleSheet('background-color: black')

        #set layout
        baseLayout = QVBoxLayout()
        widget.setLayout(baseLayout)
        
        #set picture and chart layout
        pclayout = QHBoxLayout()
        baseLayout.addLayout(pclayout, stretch=2)

        #set buttons layout
        btlayout = QGridLayout()
        baseLayout.addLayout(btlayout, stretch=1)

        #set layout for radiobtns
        rdbuttonVlayout = QVBoxLayout()
        rdbuttonVlayout2 = QVBoxLayout()
        btlayout.addLayout(rdbuttonVlayout,1,0)
        btlayout.addLayout(rdbuttonVlayout2,1,1)

        #label declaration
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        #button declaration
        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        radioBtnsGroup = QButtonGroup(widget)
        radioBtnsGroup2 = QButtonGroup(widget)
        mode = {
            "Erode":cv.MORPH_ERODE, 
            "Dilate":cv.MORPH_DILATE,
            "Open":cv.MORPH_OPEN,
            "Close":cv.MORPH_CLOSE,
            "Gradient":cv.MORPH_GRADIENT
        }

        kerShape = {
            "Rectangle": cv.MORPH_RECT,
            "Cross": cv.MORPH_CROSS,
            "Ellipse": cv.MORPH_ELLIPSE,
            "Diamonds": cv.MORPH_DIAMOND
        }

        modes = self._radiobtngenerator(radioBtnsGroup,list(mode.keys()),rdbuttonVlayout)
        kershapes = self._radiobtngenerator(radioBtnsGroup2,list(kerShape.keys()),rdbuttonVlayout2)
        radioBtnsGroup.setExclusive(True)
        radioBtnsGroup2.setExclusive(True)

        #set label to it's layout
        pclayout.addWidget(label_img,Qt.AlignmentFlag.AlignCenter)
        pclayout.addWidget(label_chart,Qt.AlignmentFlag.AlignCenter)

        #set button to it's layout
        btlayout.addWidget(button_load,1,3,alignment=Qt.AlignmentFlag.AlignTop)
        btlayout.addWidget(button_refresh,1,4,alignment=Qt.AlignmentFlag.AlignTop)

        #define function onclick refresh
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))

        #define function onclick load button
        def loadFromAnotherProcess(process:str):
            if process == '':
                return
            data = getattr(self,process)
            if not data: 
                return
            self.widgetdata[widget]['reference'] = self.widgetdata[data]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[data]['path']
        button_load.currentTextChanged.connect(loadFromAnotherProcess)

        #define function onclick radiobtn
        def showonclick(arg1:str, arg2:str):
            ref:Img = self.widgetdata[widget]['reference']
            if ref is None or arg1 == '' or arg2 == '':
                label_img.setText('ambil gambar dari proses lain terlebih dahulu')       
                return
            
            self.widgetdata[widget]['img'] = Img(ref.morphology(mode[arg1],kerShape[arg2]))

            pixmap1 = self._cv2_to_pixmap(self.widgetdata[widget]['img'].img)
            buff = Plot.makePlot(self.widgetdata[widget]['img'].img).getBuf()

            qimg = QImage.fromData(buff.read())
            pixmap2 = QPixmap.fromImage(qimg)

            self._display_to_label(label=label_img,pic=pixmap1)
            self._display_to_label(label_chart, pixmap2)

        radioBtnsGroup.buttonClicked.connect(
            lambda btn: showonclick(btn.text(), radioBtnsGroup2.checkedButton().text() if radioBtnsGroup2.checkedButton() else "")
        )
        radioBtnsGroup2.buttonClicked.connect(
            lambda btn2: showonclick(radioBtnsGroup.checkedButton().text() if radioBtnsGroup.checkedButton() else "", btn2.text())
        )
 
    def _handlesegmentation(self, widget:QWidget):
        self.widgetdata[widget] ={'img': None, 'path':'', 'reference': None}
        widget.setStyleSheet('background-color: black')

        #setLayout
        baseLayout = QVBoxLayout()
        pictureLayout = QHBoxLayout()
        bottomLayout = QHBoxLayout()
        samplesBtlayout = QVBoxLayout()

        #setup nesting layout
        widget.setLayout(baseLayout)
        baseLayout.addLayout(pictureLayout)
        baseLayout.addLayout(bottomLayout)
        bottomLayout.addLayout(samplesBtlayout)

        #thelabel img
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        pictureLayout.addWidget(label_img, alignment=Qt.AlignmentFlag.AlignCenter)

        #the button 
        groupBtn = QButtonGroup()
        groupBtn.setParent(widget)
        groupBtn.setExclusive(True)
        buttons = self._radiobtngenerator(groupBtn,["eccentricity","metric","sizeSegmentation"],samplesBtlayout)
        
        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')
        bottomLayout.addWidget(button_load)

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)
        bottomLayout.addWidget(button_refresh)

        #action
        def loadFromAnotherProcess(process:str):
            if process == '':
                return
            data = getattr(self,process)
            if not data: 
                return
            self.widgetdata[widget]['reference'] = self.widgetdata[data]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[data]['path']

        def showonclick(ind:str):
            ref:Img = self.widgetdata[widget]['reference']
            orig = self.widgetdata[widget]
            if ref is None or ind == '':
                label_img.setText('ambil gambar dari proses lain terlebih dahulu')      
                return
            
            match ind:
                case "eccentricity": orig['img'] = Img(ref.eccentricity())
                case "metric": orig['img'] = Img(ref.metric())
                case "sizeSegmentation": orig['img'] = Img(ref.sizeSegmentation())

            self._display_to_label(label_img, self._cv2_to_pixmap(orig['img'].img))

            

        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))
        button_load.currentTextChanged.connect(loadFromAnotherProcess)
        groupBtn.buttonClicked.connect(lambda bt: showonclick(bt.text()))

    def _handlehsvMask(self, widget:QWidget):
        self.widgetdata[widget] ={'img': None, 'path':'', 'reference': None}
        widget.setStyleSheet('background-color: black')

        #setLayout
        baseLayout = QVBoxLayout()
        pictureLayout = QHBoxLayout()
        bottomLayout = QHBoxLayout()
        samplesBtlayout = QGridLayout()

        #setup nesting layout
        widget.setLayout(baseLayout)
        baseLayout.addLayout(pictureLayout)
        baseLayout.addLayout(bottomLayout)
        bottomLayout.addLayout(samplesBtlayout)

        #thelabel img
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        pictureLayout.addWidget(label_img, alignment=Qt.AlignmentFlag.AlignCenter)

        #the button 
        groupBtn = QButtonGroup()
        groupBtn.setParent(widget)
        groupBtn.setExclusive(True)
        buttons = {}

        for k,_ in Img.HSVMASKCOLORRANGE.items():
            k = k.lower()
            if k in ("redup", "reddown"): k = "red"
            if k in buttons: continue

            bt = QPushButton()
            bt.setText(k)
            bt.setStyleSheet('color:black; background-color:' + k)

            groupBtn.addButton(bt)

            buttons[k] = bt
            n = len(buttons) - 1
            samplesBtlayout.addWidget(bt,n//3,n%3)
        
        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')
        bottomLayout.addWidget(button_load)

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)
        bottomLayout.addWidget(button_refresh)

        #action
        def loadFromAnotherProcess(process:str):
            if process == '':
                return
            data = getattr(self,process)
            if not data: 
                return
            self.widgetdata[widget]['reference'] = self.widgetdata[data]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[data]['path']

        def showonclick(ind:str):
            ref:Img = self.widgetdata[widget]['reference']
            orig = self.widgetdata[widget]
            if ref is None or ind == '':
                label_img.setText('ambil gambar dari proses lain terlebih dahulu')      
                return
            
            bw =  Img(ref.hsvMask(ind))

            if not bw or not bw.img:
                return
            orig['img'] = Img((bw.img * 255).astype(np.uint8))

            dpl = (bw.img[...,np.newaxis] *  ref.img).clip(0,255).astype(np.uint8)
            self._display_to_label(label_img, self._cv2_to_pixmap(dpl))

            

        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[3]))
        button_load.currentTextChanged.connect(loadFromAnotherProcess)
        groupBtn.buttonClicked.connect(lambda bt: showonclick(bt.text()))

    def _handlepoint(self, widget:QWidget):
        self.widgetdata[widget] ={'img': None, 'path':'', 'ref': None ,'source': None}
        
        widget.setStyleSheet('background-color: black')

        #layout 
        mainlayout = QVBoxLayout()
        widget.setLayout(mainlayout)

        upperlayout = QHBoxLayout()
        lowerlayout = QHBoxLayout()
        mainlayout.addLayout(upperlayout,1)
        mainlayout.addLayout(lowerlayout,1)

        bckg_layout = QVBoxLayout()
        lowerlayout.addLayout(bckg_layout)
        
        #image
        label_img = ClickLable()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        upperlayout.addWidget(label_img,alignment=Qt.AlignmentFlag.AlignCenter)

        #load from another process button
        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')
        lowerlayout.addWidget(button_load)

        #refresh button
        refresh = QPushButton()
        refresh.setText('refresh')
        refresh.setStyleSheet('color: white')
        lowerlayout.addWidget(refresh)

        #background
        bgGroup = QButtonGroup(parent=widget)
        bgGroup.setExclusive(True)
        listbutton = self._radiobtngenerator(bgGroup,["original","black"],bckg_layout)

        #action
        refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))

        def loadFromAnotherProcess(process:str):
            if process == '':
                return
            data = getattr(self,process)
            if not data: 
                return
            self.widgetdata[widget]['source'] = self.widgetdata[data]['img']
            self.widgetdata[widget]['ref'] = self.widgetdata[data]['img'].pointExtractor()
            self.widgetdata[widget]['path'] = self.widgetdata[data]['path']
            self.widgetdata[widget]['img'] = None


        button_load.currentTextChanged.connect(loadFromAnotherProcess)

        twopoints = deque(maxlen=2) #accumulator

        def showpoints(opt: str):
            datas:dict = self.widgetdata[widget]
            twopoints.clear()

            if datas['ref'] is None:
                return
            
            points = datas['ref']['getPoints']
            default = datas['ref']['drawPoints'](points)
            
            if opt == 'original' and datas['source'].img.shape == default.shape :
                mask = (default > 0)
                bgAndPoints = (datas['source'].img.copy() * 0.5).astype(np.uint8)

                bgAndPoints[mask] = default[mask]
                datas['img'] = Img(bgAndPoints)
            else:
                datas['img'] = Img(default)
            
            self._display_to_label(label_img,self._cv2_to_pixmap(datas['img'].img))
        
        bgGroup.buttonClicked.connect(lambda bt: showpoints(bt.text()))


        def pointToPoint(tp: tuple[int,int]):
            x, y = tp
            datas = self.widgetdata[widget]

            if x < 0 or y < 0 or datas['ref'] is None or datas['img'] is None:
                return
            
            pts = datas['ref']['getPoints']
            if pts is None or len(pts) == 0:
                return

            diff = pts - np.array([x, y])
            dist2 = np.sum(diff**2, axis=1)
            idx = np.argmin(dist2)

            twopoints.append(tuple(pts[idx]))

            if len(twopoints) < 2:
                return
            
            obj = datas['ref']
            pA, pB = twopoints

            distance = obj['getDistance'](pA,pB)
            degree = obj['getDegree'](pA,pB)
            
            startP = ((pA[0] + pB[0])//2, (pA[1] + pB[1])//2)
            
            image = obj['drawLine'](datas['img'].img,pA,pB)
            image = obj['drawText'](image, [f'dis={distance:.2f} px', f'deg={degree:.2f}'],startP)

            self._display_to_label(label_img,self._cv2_to_pixmap(image))

        label_img.clicked.connect(lambda a,b: pointToPoint(self._gui_to_img_scaler(a,b,label_img)))


        

    def _handleconvolv(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path':'', 'reference': None}
        widget.setStyleSheet('background-color: black')

        #set layout
        baseLayout = QVBoxLayout()
        widget.setLayout(baseLayout)
        
        #set picture and chart layout
        pclayout = QHBoxLayout()
        baseLayout.addLayout(pclayout, stretch=2)

        #set buttons layout
        btlayout = QGridLayout()
        baseLayout.addLayout(btlayout, stretch=1)

        #set layout for radiobtns
        rdbuttonVlayout = QVBoxLayout()
        btlayout.addLayout(rdbuttonVlayout,1,0)

        #label declaration
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        #button declaration
        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        radioBtnsGroup = QButtonGroup(widget)
        radiobtns = self._radiobtngenerator(radioBtnsGroup,['lp1', 'lp2', 'lp3', 'hp1','hp2','hp3'],rdbuttonVlayout)
        radioBtnsGroup.setExclusive(True)

        #set label to it's layout
        pclayout.addWidget(label_img,Qt.AlignmentFlag.AlignCenter)
        pclayout.addWidget(label_chart,Qt.AlignmentFlag.AlignCenter)

        #set button to it's layout
        btlayout.addWidget(button_load,1,1,alignment=Qt.AlignmentFlag.AlignTop)
        btlayout.addWidget(button_refresh,1,2,alignment=Qt.AlignmentFlag.AlignTop)

        #define function onclick refresh
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))

        #define function onclick load button
        def loadFromAnotherProcess(process:str):
            if process == '':
                return
            data = getattr(self,process)
            if not data: 
                return
            self.widgetdata[widget]['reference'] = self.widgetdata[data]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[data]['path']
        button_load.currentTextChanged.connect(loadFromAnotherProcess)

        #define function onclick radiobtn
        def showonclick(ind:str):
            ref:Img = self.widgetdata[widget]['reference']
            if ref is None or ind == '':
                label_img.setText('ambil gambar dari proses lain terlebih dahulu')       
                return

            match ind:
                case 'lp1': self.widgetdata[widget]['img'] = Img(ref.convolv(Img.kernelLowpass1))
                case 'lp2': self.widgetdata[widget]['img'] = Img(ref.convolv(Img.kernelLowpass2))
                case 'lp3': self.widgetdata[widget]['img'] = Img(ref.convolv(Img.kernelLowpass3))
                case 'hp1': self.widgetdata[widget]['img'] = Img(ref.convolv(Img.kernelHighpass1))
                case 'hp2': self.widgetdata[widget]['img'] = Img(ref.convolv(Img.kernelHighpass2))
                case 'hp3': self.widgetdata[widget]['img'] = Img(ref.convolv(Img.kernelHighpass3))
                case _:return

            pixmap1 = self._cv2_to_pixmap(self.widgetdata[widget]['img'].img)
            buff = Plot.makePlot(self.widgetdata[widget]['img'].img).getBuf()

            qimg = QImage.fromData(buff.read())
            pixmap2 = QPixmap.fromImage(qimg)

            self._display_to_label(label=label_img,pic=pixmap1)
            self._display_to_label(label_chart, pixmap2)

        radioBtnsGroup.buttonClicked.connect(lambda btn: showonclick(btn.text()))

    def _handlecontrast(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path':'', 'reference': None}
        widget.setStyleSheet('background-color: black')

        #set layout
        baseLayout = QVBoxLayout()
        widget.setLayout(baseLayout)
        
        #set picture and chart layout
        pclayout = QHBoxLayout()
        baseLayout.addLayout(pclayout, stretch=2)

        #set buttons layout
        btlayout = QGridLayout()
        baseLayout.addLayout(btlayout, stretch=1)

        #set layout for radiobtns
        rdbuttonVlayout = QVBoxLayout()
        btlayout.addLayout(rdbuttonVlayout,1,0)

        #label declaration
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        #button declaration
        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        radioBtnsGroup = QButtonGroup(widget)
        radiobtns = self._radiobtngenerator(radioBtnsGroup,['stretch pixel', 'Adjust contrast'],rdbuttonVlayout)
        radioBtnsGroup.setExclusive(True)

        #create slider for contrastadjust
        hSlider = QSlider()
        hSlider.setRange(0,100)    
        hSlider.setOrientation(Qt.Orientation.Horizontal)
        hSlider.setFixedSize(0,0)
        hSlider.setVisible(False)

        #set label to it's layout
        pclayout.addWidget(label_img)
        pclayout.addWidget(label_chart)

        #set button to it's layout
        btlayout.addWidget(hSlider,0,0)
        btlayout.addWidget(button_load,1,1,alignment=Qt.AlignmentFlag.AlignTop)
        btlayout.addWidget(button_refresh,1,2,alignment=Qt.AlignmentFlag.AlignTop)

        #define function onclick refresh
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))

        #define function onclick load button
        def loadFromAnotherProcess(process:str):
            if process == '':
                return
            data = getattr(self,process)
            if not data: 
                return
            self.widgetdata[widget]['reference'] = self.widgetdata[data]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[data]['path']
        button_load.currentTextChanged.connect(loadFromAnotherProcess)

        #define function onclick radiobtn
        def showonclick(ind:str, contrastval:float):
            ref:Img = self.widgetdata[widget]['reference']
            if ref is None or ind == '':
                label_img.setText('ambil gambar dari proses lain terlebih dahulu')       
                return

            match ind:
                case 'stretch pixel': 
                    self.widgetdata[widget]['img'] = Img(ref.stretchPixelDist())
                    hSlider.setVisible(False)
                case 'Adjust contrast': 
                    self.widgetdata[widget]['img'] = Img(ref.contrastAdjust(alpha=contrastval))
                    hSlider.setFixedSize(int(widget.width()*0.2),10)
                    hSlider.setVisible(True)

            pixmap1 = self._cv2_to_pixmap(self.widgetdata[widget]['img'].img)
            buff = Plot.makePlot(self.widgetdata[widget]['img'].img).getBuf()

            qimg = QImage.fromData(buff.read())
            pixmap2 = QPixmap.fromImage(qimg)

            self._display_to_label(label=label_img,pic=pixmap1)
            self._display_to_label(label_chart, pixmap2)

        radioBtnsGroup.buttonClicked.connect(lambda btn: showonclick(btn.text(),1.5))
        hSlider.sliderReleased.connect(lambda : showonclick('Adjust contrast',hSlider.value()/10))
        hSlider.valueChanged.connect(lambda v: QToolTip.showText(QCursor.pos(),str(v/10)))

    def _handlegrayscale(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path': ''}

        widget.setStyleSheet('background-color: black')

        vlayout = QVBoxLayout()
        widget.setLayout(vlayout)

        # container untuk gambar dan chart
        container = QWidget()
        container2 = QWidget()

        cont2layout = QHBoxLayout()
        Himglayout = QGridLayout()

        container.setLayout(Himglayout)
        container2.setLayout(cont2layout)

        # isi container
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        Himglayout.addWidget(label_img,0,0)
        Himglayout.addWidget(label_chart,0,1)
        Himglayout.setSpacing(2)
        Himglayout.setContentsMargins(0,0,0,0)

        # tombol proses
        button_proses = QPushButton('Load Img from file')
        button_proses.setStyleSheet('background-color: dark gray')

        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        cont2layout.addWidget(button_proses)
        cont2layout.addWidget(button_load)
        cont2layout.addWidget(button_refresh,1)
       

        # tambahkan ke layout utama
        vlayout.addWidget(container, stretch=2, alignment=Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(container2,stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

        def loadonclick(arg:Img | None):
            if not self.widgetdata[widget]['path']:
                label_img.setText('Load file terlebih dahulu')
                label_img.setStyleSheet('background-color: red')
                return
            
            datadict = self.widgetdata[widget]
            #try to load file
            try:
                if isinstance(arg, Img):
                    datadict['img'] = Img(arg.toGrayscale())
                else:
                    datadict['img'] = Img(Img.makeTemp(datadict['path']).toGrayscale())

                self._display_to_label(label_img,self._cv2_to_pixmap(datadict['img'].img))
                buf = Plot.makePlot(datadict['img'].img).getBuf()

            # ubah buffer → QPixmap
                qimg = QImage.fromData(buf.read())
                pixmap = QPixmap.fromImage(qimg)
                self._display_to_label(label_chart, pixmap)

            except Exception as e:
                label_img.setText(f'err :{e}')
            
        def helper(text):
            if not text:
                return
            wd = getattr(self,text)
            im = self.widgetdata[wd]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[wd]['path']
            loadonclick(im)


        button_proses.clicked.connect(loadonclick)
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))
        button_load.currentTextChanged.connect(helper)

    def _handleBW (self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path': '', 'ref': None}

        widget.setStyleSheet('background-color: black')

        vlayout = QVBoxLayout()
        widget.setLayout(vlayout)

        # container untuk gambar dan chart
        container = QWidget()
        container2 = QWidget()

        cont2layout = QHBoxLayout()
        Himglayout = QGridLayout()

        container.setLayout(Himglayout)
        container2.setLayout(cont2layout)

        # isi container
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        Himglayout.addWidget(label_img,0,0)
        Himglayout.addWidget(label_chart,0,1)
        Himglayout.setSpacing(2)
        Himglayout.setContentsMargins(0,0,0,0)
        
        #tambahkan slider pembersih objek untuk image segmentation
        slider = QSlider()
        slider.setRange(1,100)
        slider.setSingleStep(1)
        slider.setHidden(True)

        # tombol proses
        button_proses = QPushButton('Load Img from file')
        button_proses.setStyleSheet('background-color: dark gray')

        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        cont2layout.addWidget(slider)
        cont2layout.addWidget(button_proses)
        cont2layout.addWidget(button_load)
        cont2layout.addWidget(button_refresh,1)
       

        # tambahkan ke layout utama
        vlayout.addWidget(container, stretch=2, alignment=Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(container2,stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

        def loadonclick(arg):
            if not self.widgetdata[widget]['path']:
                label_img.setText('Load file terlebih dahulu')
                label_img.setStyleSheet('background-color: red')
                return
            
            datadict = self.widgetdata[widget]
            #try to load file
            try:
                if isinstance(arg, Img):
                    datadict['img'] = Img(arg.toBW())
                else:
                    datadict['img'] = Img(Img.makeTemp(datadict['path']).toBW())

                datadict['ref'] = None
                self._display_to_label(label_img,self._cv2_to_pixmap(datadict['img'].img))
                buf = Plot.makePlot(datadict['img'].img).getBuf()

            # ubah buffer → QPixmap
                qimg = QImage.fromData(buf.read())
                pixmap = QPixmap.fromImage(qimg)
                self._display_to_label(label_chart, pixmap)
                slider.setHidden(False)
                slider.setFixedWidth(int(widget.width()*0.2))

            except Exception as e:
                label_img.setText(f'err :{e}')
            
        def helper(text):
            if not text:
                return
            wd = getattr(self,text)
            im = self.widgetdata[wd]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[wd]['path']
            loadonclick(im)

        def slide(sizeobj:int):
            datadict = self.widgetdata[widget] 

            if datadict['ref'] is None:
                datadict['ref'] = datadict['img']

            datadict['img'] = Img(datadict['ref'].clean_binary_mask(sizeobj * 10))
            self._display_to_label(label_img,self._cv2_to_pixmap(datadict['img'].img))


        button_proses.clicked.connect(loadonclick)
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))
        button_load.currentTextChanged.connect(helper)
        slider.sliderReleased.connect(lambda :slide(slider.value()))

    def _handleneg(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path': ''}

        widget.setStyleSheet('background-color: black')

        vlayout = QVBoxLayout()
        widget.setLayout(vlayout)

        # container untuk gambar dan chart
        container = QWidget()
        container2 = QWidget()

        cont2layout = QHBoxLayout()
        Himglayout = QGridLayout()

        container.setLayout(Himglayout)
        container2.setLayout(cont2layout)

        # isi container
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        Himglayout.addWidget(label_img,0,0)
        Himglayout.addWidget(label_chart,0,1)
        Himglayout.setSpacing(2)
        Himglayout.setContentsMargins(0,0,0,0)

        # tombol proses
        button_proses = QPushButton('Load Img from file')
        button_proses.setStyleSheet('background-color: dark gray')

        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        cont2layout.addWidget(button_proses)
        cont2layout.addWidget(button_load)
        cont2layout.addWidget(button_refresh,1)
       

        # tambahkan ke layout utama
        vlayout.addWidget(container, stretch=2, alignment=Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(container2,stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

        def loadonclick(arg):
            if not self.widgetdata[widget]['path']:
                label_img.setText('Load file terlebih dahulu')
                label_img.setStyleSheet('background-color: red')
                return
            
            datadict = self.widgetdata[widget]
            #try to load file
            try:
                if isinstance(arg, Img):
                    datadict['img'] = Img(arg.toNeg())
                else:
                    datadict['img'] = Img(Img.makeTemp(datadict['path']).toNeg())

                self._display_to_label(label_img,self._cv2_to_pixmap(datadict['img'].img))
                buf = Plot.makePlot(datadict['img'].img).getBuf()

            # ubah buffer → QPixmap
                qimg = QImage.fromData(buf.read())
                pixmap = QPixmap.fromImage(qimg)
                self._display_to_label(label_chart, pixmap)

            except Exception as e:
                label_img.setText(f'err :{e}')
            
        def helper(text):
            if not text:
                return
            wd = getattr(self,text)
            im = self.widgetdata[wd]['img']
            self.widgetdata[widget]['path'] = self.widgetdata[wd]['path']
            loadonclick(im)


        button_proses.clicked.connect(loadonclick)
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[2,3]))
        button_load.currentTextChanged.connect(helper)

    def _handleRGB(self, widget:QWidget):
        self.widgetdata[widget] = {'img': None, 'path': ''}
        
        widget.setStyleSheet('background-color: black')

        vlayout = QVBoxLayout()
        widget.setLayout(vlayout)

        # container untuk gambar dan chart
        container = QWidget()
        container2 = QWidget()

        cont2layout = QHBoxLayout()
        Himglayout = QGridLayout()

        container.setLayout(Himglayout)
        container2.setLayout(cont2layout)

        # isi container
        label_img = QLabel()
        label_img.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        label_chart = QLabel()
        label_chart.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        Himglayout.addWidget(label_img,0,0)
        Himglayout.addWidget(label_chart,0,1)
        Himglayout.setSpacing(2)
        Himglayout.setContentsMargins(0,0,0,0)

        #radiobutton rgb
        rgbvlayout = QVBoxLayout()
        rgbOptionGroup = QButtonGroup(widget)        
        rgbOptionGroup.setExclusive(True)
        radiobtns = self._radiobtngenerator(rgbOptionGroup,['r','g','b'],rgbvlayout)

        button_load = QComboBox()
        button_load.setPlaceholderText('load dari proses lain')
        button_load.setStyleSheet('color: white')

        button_refresh = QPushButton('refresh')
        button_refresh.setStyleSheet('background-color: dark gray')
        button_refresh.setMaximumWidth(50)

        # cont2layout.addWidget(rgbOptionGroup)
        cont2layout.addLayout(rgbvlayout)
        cont2layout.addWidget(button_load)
        cont2layout.addWidget(button_refresh,1)
       

        # tambahkan ke layout utama
        vlayout.addWidget(container, stretch=2, alignment=Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(container2,stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

        def loadonclick(arg, arg2 = 'r'):
            if not self.widgetdata[widget]['path']:
                label_img.setText('Load file terlebih dahulu')
                label_img.setStyleSheet('background-color: red')
                return
            
            datadict = self.widgetdata[widget]
            #try to load file
            try:
                if isinstance(arg, Img):
                    datadict['img'] = Img(arg.dissableexcept(arg2))
                else:
                    datadict['img'] = Img(Img.makeTemp(datadict['path']).dissableexcept(arg2))

                self._display_to_label(label_img,self._cv2_to_pixmap(datadict['img'].img))
                buf = Plot.makePlot(datadict['img'].img).getBuf()

            # ubah buffer → QPixmap
                qimg = QImage.fromData(buf.read())
                pixmap = QPixmap.fromImage(qimg)
                self._display_to_label(label_chart, pixmap)

            except Exception as e:
                label_img.setText(f'err :{e}')
            
        def helper(text):
            if not text:
                return
            wd = getattr(self,text)
            im = self.widgetdata[widget]
            
            im['reference'] = self.widgetdata[wd]['img']
            im['path'] = self.widgetdata[wd]['path']
            
        rgbOptionGroup.buttonClicked.connect(lambda btn: loadonclick(self.widgetdata[widget]['reference'],btn.text()))
        button_refresh.clicked.connect(lambda: self._comboadder(widget,button_load,[3]))
        button_load.currentTextChanged.connect(helper)
        
    #menu bar handler
    def _handlefile(self, button:QPushButton):

        def onclick():
            widget = cast(QTabWidget, self.centralWidget()).currentWidget()

            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                'Pilih Gambar', 
                '', 
                'Images (*.png *.jpg *.jpeg *.bmp *.tiff)'
            )

            if file_path:
                self.widgetdata[widget]['path'] = file_path
                button.setStyleSheet('background-color: green')
            else:
                button.setStyleSheet('background-color: red')


        button.clicked.connect(onclick)
    
    def _cv2_to_pixmap(self, img:cv.Mat) -> QPixmap:
        """Konversi OpenCV image (BGR/Gray) ke QPixmap"""
        if len(img.shape) == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = img.shape
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    
    def _display_to_label(self, label:QLabel | ClickLable, pic:QPixmap):
        width = self.width()//2
        height = self.height()//2
        
        w = int(width * 0.1)
        h = int(height * 0.1)
        scaled_pixmap = pic.scaled(
            width - w,
            height - h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)

    def _comboadder(self,widget:QWidget, Combbutton: QComboBox, shapes: list[int]):
        Combbutton.clear()
        for name in self.options:
            widg = getattr(self,name)
            image = self.widgetdata[widg]['img']
            if image and (len(image.img.shape) in shapes) and widg != widget:
                Combbutton.addItem(name) 
                Combbutton.setStyleSheet('color: white')

    def _radiobtngenerator(self, groupbtn: QButtonGroup, btns: list[str], layout: QHBoxLayout | QVBoxLayout) -> list[QRadioButton]:
        radios = []
        for text in btns:
            rb = QRadioButton(text)
            rb.setStyleSheet('color: white')
            groupbtn.addButton(rb)
            layout.addWidget(rb)
            radios.append(rb)
        return radios
    
    def _gui_to_img_scaler(self, mx:int, my:int, label: ClickLable) -> tuple[int,int]:
        widget = cast(QTabWidget, self.centralWidget()).currentWidget()
        datas:dict = self.widgetdata[widget]

        if datas["img"] is None: 
            return (-1,-1)
        
        img_h, img_w = datas["img"].img.shape[:2]

        gui_w = label.width()
        gui_h = label.height()

        sx = img_w / gui_w
        sy = img_h / gui_h

        return int(mx * sx), int(my * sy)


app = QApplication(sys.argv)
window = Mainwindow()
window.show()
app.exec()

