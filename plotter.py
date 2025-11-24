 # ---- tampilkan histogram di label_chart ---- // To DO : rapikan
from __future__ import annotations
import matplotlib.pyplot as plt
import io
import cv2 as cv
import numpy as np
from PyQt6.QtGui import QPixmap, QImage


# buat figure matplotlib
class Plot:
    def __init__(self,data) -> None:
        self.buf = io.BytesIO()

        if data.ndim == 3:  # RGB / BGR
            color = ('b', 'g', 'r')
            fig, ax = plt.subplots(3,1, figsize=(9, 3), sharex=True)
            # pastikan ax selalu iterable (array)
            ax = np.atleast_1d(ax)
            for i, col in enumerate(color):
                hist = cv.calcHist([data], [i], None, [256], [0, 256]).ravel()
                ax[i].bar(np.arange(256),hist, color=col)
                ax[i].set_xlim([0, 256])
                ax[i].set_ylabel(col.upper())
        else:  # grayscale
            fig, ax = plt.subplots(figsize=(4, 3))
            hist = cv.calcHist([data], [0], None, [256], [0, 256]).ravel()
            ax.bar(np.arange(256),hist, color='k')
            ax.set_xlim((0, 256))
            ax.set_title("One Channel")

        # fig.tight_layout()
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
        

