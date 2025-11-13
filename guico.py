from __future__ import annotations
import cv2 as cv
import numpy as np


class Img:

    supportedExtention = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff"})
    __slots__ = ["img"]

    def __init__(self, p: str | np.ndarray) -> None:

        self.img = None
        if isinstance(p, np.ndarray):
            if p.ndim not in (2, 3):
                raise ValueError("invalid image array shape")
            self.img = p.copy()
            return

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
            return self.img
        ## Weighted formula: Gray = 0.2989*R + 0.5870*G + 0.1140*B -- important !! cv2 using BGR format
        gray = (
            0.1140 * self.img[:, :, 0]
            + 0.5870 * self.img[:, :, 1]
            + 0.2989 * self.img[:, :, 2]
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
                return np.dstack([filler, filler, self.img[:, :, 2]])
            case 'g':
                return np.dstack([filler, self.img[:, :, 1], filler])
            case 'b':
                return np.dstack([self.img[:, :, 0], filler, filler])
            case _:
                return None

    @staticmethod
    def to3Channel(obj: Img) -> bool:
        if obj.img is None or len(obj.img.shape) != 2:
            return False
        obj.img = np.stack((obj.img, obj.img, obj.img), axis=-1)

        return obj.img is not None


if __name__ == "__main__":
    print('this code section is for testing only')
    inp = input("masukkan path").strip()
    mg = Img(inp)

    w = mg.dissableexcept('g')

    cv.namedWindow("bgr", cv.WINDOW_NORMAL)
    cv.imshow("bgr", w)
    cv.waitKey(0)
    cv.destroyAllWindows()
