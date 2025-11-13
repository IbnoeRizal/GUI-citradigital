from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QSizePolicy, QGridLayout,
                              QMainWindow, QHBoxLayout, QVBoxLayout, QTabWidget, QLabel, QComboBox, QRadioButton, QButtonGroup)
from PyQt6.QtGui import (QImage, QPixmap)
from typing import cast

from plotter import Plot
from guico import Img

import sys
import cv2 as cv

class Mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.options = ('original','grayscale', 'BW', 'neg', 'RGB')
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
                orig = arg if isinstance(arg,Img) else Img(datadict['path'])
                orig.img = orig.toGrayscale()

                datadict['img'] = orig
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
        button_refresh.clicked.connect(lambda: self._combobadder(widget,button_load,[2,3]))
        button_load.currentTextChanged.connect(helper)

    def _handleBW (self, widget:QWidget):
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
                orig = arg if isinstance(arg,Img) else Img(datadict['path'])
                orig.img = orig.toBW()

                datadict['img'] = orig
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
        button_refresh.clicked.connect(lambda: self._combobadder(widget,button_load,[2,3]))
        button_load.currentTextChanged.connect(helper)

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
                orig = arg if isinstance(arg,Img) else Img(datadict['path'])
                orig.img = orig.toNeg()

                datadict['img'] = orig
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
        button_refresh.clicked.connect(lambda: self._combobadder(widget,button_load,[2,3]))
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
                orig = arg if isinstance(arg,Img) else Img(datadict['path'])
                orig.img = orig.dissableexcept(arg2)
                datadict['img'] = orig
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

        rgbOptionGroup.buttonClicked.connect(lambda btn: loadonclick(None,btn.text()))
        button_refresh.clicked.connect(lambda: self._combobadder(widget,button_load,[3]))
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
    
    def _display_to_label(self, label:QLabel, pic:QPixmap):
        width = self.width()//2
        height = self.height()//2
        scaled_pixmap = pic.scaled(
            width,
            height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)

    def _combobadder(self,widget:QWidget, Combbutton: QComboBox, shapes: list[int]):
        Combbutton.clear()
        for name in self.options:
            widg = getattr(self,name)
            image = self.widgetdata[widg]['img']
            if image and (len(image.img.shape) in shapes) and widg != widget:
                Combbutton.addItem(name) 

    def _radiobtngenerator(self, groupbtn: QButtonGroup, btns: list[str], layout: QVBoxLayout) -> list[QRadioButton]:
        radios = []
        for text in btns:
            rb = QRadioButton(text)
            groupbtn.addButton(rb)
            layout.addWidget(rb)
            radios.append(rb)
        return radios


app = QApplication(sys.argv)
window = Mainwindow()
window.show()
app.exec()

