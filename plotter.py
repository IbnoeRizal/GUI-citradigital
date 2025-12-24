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

        return self
    
    def heapMap(self,texts:np.ndarray) -> Plot:
        if self.img is None or self.img.ndim != 3:
            return self
        
        n = len(texts)

        if n != 0 and not isinstance(texts[0],dict):
            return self
        
        fig, ax = plt.subplots(n or 1, 2 if n else 1, sharex='col',figsize=(10, 3*n if n else 3))
        ax = np.atleast_1d(ax)
        
        
        for i in range(len(self.img)):
            Z = self.img[i]

            ax[i,0].imshow(Z)
            if 'angle' in texts[i]:
                ax[i,0].set_title(f'angle: {int(texts[i]['angle'])}')
            
            if n:
                ax[i,1].axis('off')
                text_content = '\n'.join([f'{k}: {v:.3f}' for k, v in texts[i].items() if k != 'angle'])
                ax[i, 1].text(0.5, 0.5, text_content, 
                         transform=ax[i, 1].transAxes,
                         fontsize=11,
                         verticalalignment='center',
                         horizontalalignment='left',
                         family='monospace')
                
        fig.tight_layout()
        plt.savefig(self.buf, format='png')
        plt.close(fig)
        self.buf.seek(0)
        
        return self
        

