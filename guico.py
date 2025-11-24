from __future__ import annotations
import cv2 as cv
import numpy as np
from numpy._core.numeric import ndarray


class Img:

    supportedExtention = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff"})
    __slots__ = ["img"]
    kernelLowpass1 = np.array([[1/16, 1/8, 1/16],
                               [1/8, 1/4, 1/8],
                               [1/16, 1/8, 1/4]])

    kernelLowpass2 = np.array([[1/10,1/10,1/10],
                               [1/10,1/5,1/10],
                               [1/10,1/10,1/10]])

    kernelLowpass3 = np.array([[1/9,1/9,1/9],
                               [1/9,1/9,1/9],
                               [1/9,1/9,1/9]])

    kernelHighpass1 = np.array([[-1,-1,-1],
                                [-1,8,-1],
                                [-1,-1,-1]])

    kernelHighpass2 = np.array([[0,-1,0],
                                [-1,5,-1],
                                [0,-1,0]])

    kernelHighpass3 = np.array([[1,-2,1],
                                [-2,5,-2],
                                [1,-2,1]])

    def __init__(self, p: str | np.ndarray) -> None:

        self.img = None
        if isinstance(p, np.ndarray):
            if p.ndim not in (2, 3):
                raise ValueError("invalid image array shape")
            self.img = p
            return
        if not isinstance(p,str):
            raise ValueError("invalid, p(parameter) has to be either string path or img")
        p = p.strip()
        n = len(p)

        if n == 0:
            raise ValueError("path must not be empty")
        elif Img.validfile(p):
            self.img = cv.imread(p)

        if self.img is None:
            raise RuntimeError("can't read the file")

    @staticmethod
    def validfile(p: str) -> bool:
        return any(p.lower().endswith(ext) for ext in Img.supportedExtention)

    def toGrayscale(self) -> np.ndarray | None:
        if self.img is None:
            return None
        elif len(self.img.shape) == 2:
            return self.img.copy()
        ## Weighted formula: Gray = 0.2989*R + 0.5870*G + 0.1140*B -- important !! cv2 using BGR format
        cp = self.img      
        gray = (
            0.1140 * cp[:, :, 0]
            + 0.5870 * cp[:, :, 1]
            + 0.2989 * cp[:, :, 2]
        )
        return np.clip(gray, 0, 255).astype(np.uint8)

    def toBW(self, intensity: int = 127) -> np.ndarray | None:
        gr = self.toGrayscale()

        if gr is None:
            return None

        if not np.issubdtype(gr.dtype, np.integer):
            if gr.max() > gr.min():
                gr = (255 * (gr - gr.min()) / (gr.max() - gr.min())).astype(np.uint8)
            else:
                gr = np.zeros_like(gr, dtype=np.uint8)

        intensity = max(0, min(255, int(intensity)))

        _, bw = cv.threshold(gr, intensity, 255, cv.THRESH_BINARY)

        return bw

    def toNeg(self) -> np.ndarray | None:
        if self.img is None:
            return None
        return (255 - self.img).astype(np.uint8)
    
    def dissableexcept(self,select:str) -> np.ndarray | None:
        if self.img is None or self.img.ndim < 3:
            return None
        select = select.strip().lower()

        filler = np.zeros((self.img.shape[:2]),dtype=np.uint8)
        match select:
            case 'r':
                return np.dstack([filler, filler, self.img[:, :, 2].copy()])
            case 'g':
                return np.dstack([filler, self.img[:, :, 1].copy(), filler])
            case 'b':
                return np.dstack([self.img[:, :, 0].copy(), filler, filler])
            case _:
                return None

    @staticmethod
    def to3Channel(obj: Img) -> bool:
        if obj.img is None or len(obj.img.shape) != 2:
            return False
        obj.img = np.stack((obj.img, obj.img, obj.img), axis=-1)

        return obj.img is not None

    @staticmethod
    def makeTemp(path:str):
        return Img(path)

    def stretchPixelDist(self, low = 0, high = 255) -> np.ndarray | None:

        low = min(255, max(0,low))
        high = min(255, max(0, high))

        if self.img is None or self.img.ndim not in (2,3):
            return None

        def helper(ch:np.ndarray):
            v_max, v_min = np.amax(ch) , np.amin(ch)

            if v_min == v_max:
                return np.full_like(ch,low,dtype=np.uint8)
        
            perbandingan = (high - low) / (v_max - v_min)

            result:np.ndarray = low + (ch - v_min) * perbandingan
            return result.clip(min=0, max=255).astype(np.uint8)
  
        temp = np.empty(self.img.shape,dtype=np.uint8)
        
        if temp.ndim == 2:
            temp = helper(self.img)
        else:
            for i in range(3):
                temp[:,:,i] = helper(self.img[:,:,i])
        return temp

    def contrastAdjust(self, alpha=1.2) -> np.ndarray | None:
        if self.img is None:
            return None
        channel:int = self.img.ndim or 0
        match channel:
            case 2:
                avg = np.mean(self.img)
                result = alpha * (self.img - avg) + avg
                return result.clip(0,255).astype(np.uint8)
            case 3:
                h, w, _ = self.img.shape
                result = np.empty((h, w, 3), dtype=np.uint8)

                for i in range(3):
                    ch = self.img[:,:,i]
                    avg = np.mean(ch)
                    result[:,:,i] = (alpha * (ch - avg) + avg).clip(0,255)
                return result

        return None

    def convolv(self,kernelOpt:np.ndarray) -> np.ndarray | None:
        if self.img is None or kernelOpt is None:
            return None
        return cv.filter2D(self.img,-1,kernelOpt)



if __name__ == "__main__":
    print('this code section is for testing only')
    inp = input("masukkan path").strip()
    mg = Img(inp)

    w = mg.convolv(kernelOpt=Img.kernelHighpass2)

    cv.namedWindow("stretchPixelDist", cv.WINDOW_NORMAL)
    cv.imshow("stretchPixelDist", w)
    cv.waitKey(0)
    cv.destroyAllWindows()
