 # ---- tampilkan histogram di label_chart ---- // To DO : rapikan
from __future__ import annotations
import matplotlib.pyplot as plt
import io
import cv2 as cv
import numpy as np


# buat figure matplotlib
class Plot:
    def __init__(self,data) -> None:
        self.img:np.ndarray = data
        self.buf = io.BytesIO()
        

    @staticmethod
    def makePlot(data) -> Plot:
        return Plot(data)
    
    def getBuf(self):
        return self.buf
    
    def histogram(self) -> Plot:
        if self.img is None:
            return self
        
        dim:int = self.img.ndim

        if dim not in (2,3):
            return self
        
        color = ('b', 'g', 'r') if dim == 3 else ('k',)
        fig, ax = plt.subplots(len(color),1, figsize=(9, 3), sharex=True)

        ax = np.atleast_1d(ax)
        for i, col in enumerate(color):
            hist = cv.calcHist([self.img], [i], None, [256], [0, 256]).ravel()
            ax[i].bar(np.arange(256),hist, color=col)
            ax[i].set_xlim([0, 256])
        
        ax[0].set_title("One Channel" if dim == 2 else "Three Channel")

        fig.tight_layout()
        fig.align_titles()

        # simpan ke buffer
        plt.savefig(self.buf, format='png')
        plt.close(fig)
        self.buf.seek(0)

    @staticmethod
    def makePlot(data) -> Plot:
        return Plot(data)
    
    def getBuf(self):
        return self.buf
    
    def _plot(self, ax, img) -> None:
        pass
        

