from __future__ import annotations
import cv2 as cv
import numpy as np


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

    HSVMASKCOLORRANGE = {
        "redDown":  np.array([  0,   0,  14], dtype=np.uint8),   # 0–10
        "redUP":    np.array([242, 255, 255], dtype=np.uint8),   # 170–179
        "orange":   np.array([ 14,  28,  35], dtype=np.uint8),   # 10–25
        "yellow":   np.array([ 35,  49,  63], dtype=np.uint8),   # 25–35
        "green":    np.array([ 49, 121, 170], dtype=np.uint8),   # 35–85
        "cyan":     np.array([121, 142, 170], dtype=np.uint8),   # 85–100
        "blue":     np.array([142, 185, 213], dtype=np.uint8),   # 100–130
        "purple":   np.array([185, 213, 228], dtype=np.uint8),   # 130–150
        "magenta":  np.array([213, 234, 242], dtype=np.uint8),   # 150–170
        "pink":     np.array([228, 242, 255], dtype=np.uint8),   # 160–179 (soft)
    } 

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
        """
        mengubah gambar menjadi skala abu

        Weighted formula: Gray = 0.2989*R + 0.5870*G + 0.1140*B 

        return
        ______
        image : np.ndarray
            ndim(2)
        """
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
        """
        Mengubah image menjadi satu channel dengan treshold pixel 127 (default)

        Parameter
        _________
        intensity: int = 25
            nilai treshold (ambang), pixel = 0 jika nilai pixel < intensity selain itu pixel = 1

        Return
        ______
        image : np.ndarray
            ndim(2)
        """
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
        """
        mencerminkan nilai pixel 0 sampai 255

        Return
        ______
        image : np.ndarray
            ndim(3 or 2)
        """
        if self.img is None:
            return None
        return (255 - self.img).astype(np.uint8)
    
    def dissableexcept(self,select:str) -> np.ndarray | None:
        """
        menonaktifkan dua channel rgb kecuali yang dipilih
        
        Parameter
        _________
        select:str
            option (r or g or b)

        Return
        ______
        image : np.ndarray
            ndim(3)
        """
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
        """
        Membuat image dengan ndim = 2 menjadi ndim = 3

        Return
        ______
        image : np.ndarray 
            ndim(3)
        """
        if obj.img is None or len(obj.img.shape) != 2:
            return False
        obj.img = np.stack((obj.img, obj.img, obj.img), axis=-1).astype(np.uint8)

        return obj.img is not None

    @staticmethod
    def makeTemp(path:str):
        return Img(path)

    def stretchPixelDist(self, low = 0, high = 255) -> np.ndarray | None:
        """
        Menghitung persebaran nilai pixel, dan meregangkannya dari 0 ke 255 (default)
        
        Parameter
        _________
        low = 0: int
            default batas kiri peregangan

        high = 255: int
            default batas kanan peregangan

        Return
        ______
        image : np.ndarray 
            ndim(2 or 3)
        
        """
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
        """
        Menaikkan atau Menurunkan nilai pixel terhadap rata - rata tiap channel warna

        Parameter
        _________
        alpha: float
            seberapa besar penambahan jarak dari rata - rata pixel
            alpha < 0 : nilai pixel bergeser menjauhi ke kiri
            alpha = 0 : nilai pixel bergeser menjadi rata - rata nilai pixel
            alpha = 1 : nilai pixel tetap
            alpha > 1 : menjauhi persen jarak dari rata - rata pixel

        Return
        ______
            image : np.ndarray
                tergantung dimensi input 2d atau 3d
        """
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
        """
        Menjalankan konvolusi terhadap gambar dengan kernel
        
        Parameter
        _________
        kernelOpt : np.ndarray 
            kernel lowpass atau highpass tersedia sebagai variabel kelas

        Return
        ______
        filtered image : np.ndarray

        """
        if self.img is None or kernelOpt is None:
            return None
        return cv.filter2D(self.img,-1,kernelOpt)
    
    def clean_binary_mask(self, min_area: int = 500) -> np.ndarray|None:
        """
        Membersihkan biner mask:
        1. Threshold
        2. Morphology opening (hapus noise kecil)
        3. Connected component filtering (hapus area kecil sekali)

        Parameter
        ---------
        min_area : int
            Area minimum blob yang dipertahankan.

        Return
        ------
        mask : np.ndarray
            Binary image (0/255) yang bersih.
        """

        # 1. Threshold → biner 0 / 255
        mask = self.img if self.img.ndim == 2 else self.toBW()
        if mask is None: 
            return None

        # 2. Morphology opening → hapus titik/noise kecil
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

        # 3. Connected Components → hapus blob kecil
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

        # stats[:, cv.CC_STAT_AREA] berisi luas setiap area
        cleaned = np.zeros_like(mask)

        for i in range(1, num_labels):  # abaikan label 0 (background)
            area = stats[i, cv.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255

        return cleaned

    def hsvMask(self,colorToMask:str) -> np.ndarray | None:
        """
        mengambil warna dari image 3 channel, mengembalikan mask fuzzy ranges 0 to 1 terhadap image tersebut 1 channel
        
        Parameter
        _________
        colorToMask: str
            the key of Img.HSVMASKCOLORRANGE

        Return
        ______
        image: np.ndarray
            ndim(2) ranges 0.0 to 1.0
        """
        colorToMask = colorToMask.strip().lower()
        if self.img is None or not colorToMask or self.img.ndim != 3:
            return None

        hsv = cv.cvtColor(self.img,cv.COLOR_BGR2HSV)
        H = hsv[:,:,0].astype(np.float32) * (255/179) #mengambil channel pertama yaitu Hue normalisasi menjadi 0 sampai 255
        #S = hsv[:,:,1]#mengambil channel kedua yaitu saturation
        #V = hsv[:,:,2]#mengambil channel ketiga yaitu value

       
        def fuzzyTriangle(x, abc):
            left, peak, right = abc

            res = np.zeros_like(x, dtype=np.float32)

            # sisi kiri (naik)
            mask_left = (x >= left) & (x <= peak)
            res[mask_left] = (x[mask_left] - left) / (peak - left)

            # sisi kanan (turun)
            mask_right = (x > peak) & (x <= right)
            res[mask_right] = (right - x[mask_right]) / (right - peak)

            return res

        def fuzzyTrapesium(x, abc, R=True):
            left, peak, right = abc

            result = np.zeros_like(x, dtype=np.float32)

            if R:
                left_mask = (x >= left) & (x <= peak)
                result[left_mask] = (x[left_mask] - left) / (peak - left)
            else:
                right_mask = ((x >= peak) & (x <= right))
                result[right_mask] = (right - x[right_mask]) / (right - peak)


            peak_mask = ((x > peak) & (x <= right)) if R else ((x < peak) & (x >= left))
            result[peak_mask] = 1.0

            return result

      #  SV = fuzzyTrapesium(S,np.array([30, 50, 255], dtype=np.uint8),R=True) * fuzzyTriangle(V,np.array([10, 120, 230], dtype=np.uint8))

        if colorToMask == "red":
            L = fuzzyTrapesium(H,self.HSVMASKCOLORRANGE[colorToMask + "UP"],R=True)
            R = fuzzyTrapesium(H,self.HSVMASKCOLORRANGE[colorToMask + "Down"],R=False)
            return np.maximum(L,R) 

        if colorToMask in self.HSVMASKCOLORRANGE:
            return fuzzyTriangle(H, self.HSVMASKCOLORRANGE[colorToMask]) 
        else: return None

    def morphology(self, mode:cv.MorphTypes, kshape:cv.MorphShapes, n=5) -> np.ndarray | None:
        """
        return morphology of any ndimensional image

        Parameter
        _________
        mode:int
            One of [MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT, MORPH_HITMISS]
        kshape:int
            One of [MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE, MORPH_DIAMOND]            
            
        """
        if self.img is None:
            return None
        kernel = cv.getStructuringElement(kshape,(n,n))
        return cv.morphologyEx(self.img,mode,kernel)
        
    def eccentricity(self) -> np.ndarray | None:
        bw = cv.Canny(self.toGrayscale(),100,200)

        cp = np.dstack((bw,bw,bw))
        if bw is None or cp is None:
            return None
        
        #mendapatkan kontur dari semua objek
        #cv.RETR_EXTERNAL : titik hanya terluar dari objek
        #cv.CHAIN_APPROX_SIMPLE: titik yang berada di sepanjang garis diagonal atau tegak lurus dikompresi kecuali sudut
        cnts, hierarchy = cv.findContours(bw,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            if len(cnt) < 5: continue
            try:
                ellipse = cv.fitEllipse(cnt)
            except:
                continue

            (x, y), (axis1, axis2), angle = ellipse

            if np.isnan(x) or np.isnan(y) or np.isnan(axis1) or np.isnan(axis2):
                continue

            majorAxis = max(axis1, axis2)
            minorAxis = min(axis1, axis2)

            if majorAxis == 0: continue

            e:float = np.sqrt(1-(minorAxis/majorAxis)**2)
            if e < 0.6: continue

            cv.putText(cp, f"{e * 100:.1f} %",(int(x), int(y)), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,255), 2, cv.LINE_AA )
            cv.ellipse(cp, ellipse, (0,255,0), 2)
            

        return cp
    
    def metric(self,treshold = 0.5, color = (255,0,0)) -> np.ndarray | None:
        bw = self.toBW()
        if bw is None:
            return None
        cp = np.dstack((bw,bw,bw))

        treshold = max(0, min(1, treshold))
        
        contours, hierarchy = cv.findContours(bw,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) <= 4 : continue
            area = cv.contourArea(cnt)
            perimeter = cv.arcLength(cnt,True)

            if perimeter == 0: continue
            metric = 4 * np.pi * area / perimeter**2

            if metric > treshold:
                (x,y), radius = cv.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)

                cv.circle(cp,center,radius,color,2)
                cv.fillPoly(cp,[cnt],color)

        return cp
    
    def sizeSegmentation(self, areaLabel = [[100,400],[300,2000,5000],[4900,9000]], col = [[255,0,0],[0,255,0],[0,0,255]]) -> np.ndarray | None:
        bw = self.toBW()
        if bw is None:
            return None
        cp = np.dstack((bw,bw,bw))

        contour, hierarchy = cv.findContours(bw,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        def fuzzyTriangle(param, x):
            left, peak, right = param

            if x < left or x > right: 
                return 0

            l = (x - left) / (peak - left)
            r = (right - x) / (right - peak)

            return l if x <= peak else r
        
        def fuzzyTrapesium(param,x,riseRigtTop=True):
            left, right = param
            
            if riseRigtTop:
                res = (x - left) / (right - left)
            else :
                res = (right - x) / (right - left)
                
            return min(1, max(0, res)) 
        
        small, normal, big = areaLabel

        for cnt in contour:

            area = cv.contourArea(cnt)
            fuzzyvar = np.array([
                fuzzyTrapesium(small,area,False),
                fuzzyTriangle(normal,area),
                fuzzyTrapesium(big,area,True)
                ])
            
            pick = np.argmax(fuzzyvar)

            cv.fillPoly(cp,[cnt],col[pick])
        
        return cp



                    






if __name__ == "__main__":
    print('this code section is for testing only')
    inp = input("masukkan path").strip()
    mg = Img(inp)
    w = mg.sizeSegmentation()
    cv.namedWindow("ellipse", cv.WINDOW_NORMAL)
    cv.imshow("ellipse", w )
    cv.waitKey(0)
    cv.destroyAllWindows()
