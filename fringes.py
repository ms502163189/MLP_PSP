import numpy as np 
import cv2
from pathlib import Path
import random
from config import config

# synthetic fringe images
class Fringes():
    def __init__(self, config):
        self._c = config

    def generate_bg(self):
        W, H = self._c.pattern_size
        img = np.zeros([H, W])
        return img+self._c.light_value

    def generate_one(self, zeta, T, A, B):
        W, H = self._c.pattern_size
        img = np.zeros([H, W])
        x = np.linspace(0,W-1,W) if self._c.hv == "h" else np.linspace(0,H-1,H).reshape(-1,1) 
        y = A+B*np.cos(2*np.pi*x/T+zeta) 
        return img+y
    
    def generate_steps(self, T, A, B, step):
        assert step >= 3
        zeta_s = np.linspace(0, step-1, step)*2*np.pi/step
        images = list()
        for zeta in zeta_s:
            images.append(self.generate_one(zeta, T, A, B))
        return images
    
    def generate_all(self):
        images = list()

        for T, A, B, step in zip(self._c.Tp, self._c.A, self._c.B, self._c.steps):
            
            fringes = self.generate_steps(T, A, B, step)
            images.append(fringes)

        # images.append(self.generate_bg)
        return images

    def save_images(self):
        images = self.generate_all()
        root = self._c.pattern_path
        for f, fringes in enumerate(images):
            for s, fringe in enumerate(fringes):
                p = Path(root)/f"{f:0>2d}{s:0>2d}{self._c.hv}.bmp" 
                cv2.imwrite(str(p), fringe)
        img = self.generate_bg()
        p = Path(root)/f"bg.bmp" 
        cv2.imwrite(str(p), img)
        
    def save_npz_images(self,ind):
        images = self.generate_all()
        if ind < 20:
            for f, fringes in enumerate(images):
                for s, fringe in enumerate(fringes):
                    p = Path(self._c.pattern_path)/f"{ind:0>3d}{f:0>2d}{s:0>2d}{self._c.hv}.bmp" 
                    cv2.imwrite(str(p), fringe)
        np.savez(self._c.net_dir+f"train_data{ind:03d}", image1=images[0], image2=images[1], image3=images[2]) #将3个条纹保存在未压缩的.npz格式中

    def generate_phase(self):
        W, H = self._c.pattern_size
        img = np.zeros([H, W])
        x = np.linspace(0,W-1,W)
        phase = img+2*np.pi*x/self._c.Tp[0]
        return phase
    
    
class Fringes_Seg(Fringes):
    def __init__(self, config):
        super().__init__(config)
        self.init_T123()
        
    def init_T123(self):
        mT = lambda T1, T2: T1*T2/(T2-T1)
        T12 = mT(self._c.Tp[0],self._c.Tp[1])
        T23 = mT(self._c.Tp[1],self._c.Tp[2])
        T123 = mT(T12, T23)
        T1 = self._c.Tp[0]
        
        # obtain the del index
        indT = np.round(np.arange(np.round(T1))-T1/2).astype(np.int32)
        seg = np.linspace(0, T123, self._c.steps[0]+1)
        ind = (indT.reshape(1,-1)+seg.reshape(-1,1)).flatten()
        ind = np.round(ind).astype(np.int32) 
        
        mask = (ind>-1)*(ind<self._c.pattern_size[0])
        self.ind = ind[mask]        
        
    def generate_one(self, zeta, T, A, B):
        self._c.pattern_size[0] +=200
        y = super().generate_one(zeta,T,A,B)
        self._c.pattern_size[0] -=200
        y = np.delete(y, self.ind, 1)
        y = y[:,0:(self._c.pattern_size[0])]
        return y
        
    
class Fringes_Gamma(Fringes):
    def __init__(self, config):
        super().__init__(config)
    def generate_one(self, zeta, T, A, B):
        y = super().generate_one(zeta,T,A,B)
        y = 255*np.power(y/255, self._c.gamma) 
        y = np.round(y)
        return y 
    
    
class Fringes_Harmonics(Fringes):
    def __init__(self, config):
        super().__init__(config)
    def generate_one(self, zeta, T, A, B):
        W, H = self._c.pattern_size
        img = np.zeros([H, W])
        # x = np.linspace(0,W-1,W)
        x = np.linspace(0,W-1,W) if self._c.hv == "h" else np.linspace(0,H-1,H).reshape(-1,1) 
        phi = 2*np.pi*x/T+zeta
        y = A+B*np.cos(phi)
        h = self._c.C*np.cos(2*phi)+self._c.D*np.cos(3*phi)+self._c.E*np.cos(4*phi)+self._c.F*np.cos(5*phi)
        return np.round(img+y+h)
    
class Fringes_Harmonics_have_noise(Fringes):
    def __init__(self, config):
        super().__init__(config)
    def generate_one(self, zeta, T, A, B):
        W, H = self._c.pattern_size
        img = np.zeros([H, W])
        # x = np.linspace(0,W-1,W)
        x = np.linspace(0,W-1,W) if self._c.hv == "h" else np.linspace(0,H-1,H).reshape(-1,1)
#         y = np.linspace(0,H-1,H).reshape(-1,1) if self._c.hv == "h" np.linspace(0,W-1,W)
        phi = np.zeros(len(x))
        y = np.zeros(len(x))
        h = np.zeros(len(x))
        order = 0
        out_img = np.zeros([H, W])
#         aa = self._c.coefficient[0][0]
#         bb = self._c.coefficient[0][1]
#         cc = self._c.coefficient[0][2]
#         dd = self._c.coefficient[0][3]
#         ee = self._c.coefficient[0][4]
#         ff = self._c.coefficient[0][5]
        
#         for index in range(len(x)):
# #             print("index : {}".format(index))
# #             print("x[index] : {}".format(x[index]))
#             phi[index] = 2 * np.pi * x[index] / T + zeta
#             y[index] = aa + bb * np.cos(phi[index])
#             h[index] = cc * np.cos(2 * phi[index]) + dd * np.cos(3 * phi[index]) + ee * np.cos(4 * phi[index]) + ff * np.cos(5 * phi[index])
#             for i in range(int(len(self._c.y)/10)):
#                 if (y[index] + h[index] >= i * 10 and y[index] + h[index] < (i + 1) * 10):
#                     time = i
# #                     print("更新前：y[index] + h[index] : {}".format(y[index] + h[index]))
# #                     print("time : {}".format(time))
#                     break
#             a = self._c.coefficient[time][0]
#             b = self._c.coefficient[time][1]
#             c = self._c.coefficient[time][2]
#             d = self._c.coefficient[time][3]
#             e = self._c.coefficient[time][4]
#             f = self._c.coefficient[time][5]
#             y[index] = a + b * np.cos(phi[index])
#             h[index] = c * np.cos(2 * phi[index]) + d * np.cos(3 * phi[index]) + e * np.cos(4 * phi[index]) + f * np.cos(5 * phi[index])
# #             print("更新后：y[index] + h[index] : {}".format(y[index] + h[index]))
#         out_img = np.round(img + y + h)
# #         print("out_img")
# #         print(out_img)
        
#         print("len(x) : {}".format(len(x)))
#         print("H : {}".format(H))
        a = self._c.coefficient[0][0]
        b = self._c.coefficient[0][1]
        c = self._c.coefficient[0][2]
        d = self._c.coefficient[0][3]
        e = self._c.coefficient[0][4]
        f = self._c.coefficient[0][5]
        last_order = 0
        for index in range(len(x)):
            phi[index] = 2 * np.pi * x[index] / T + zeta
            for j in range(H):
                order = round(j//40)
                if (j != 0 and order != last_order):
#                 print("j : {}".format(j))
#                 print("time : {}".format(time))
                    a = self._c.coefficient[order][0]
                    b = self._c.coefficient[order][1]
                    c = self._c.coefficient[order][2]
                    d = self._c.coefficient[order][3]
                    e = self._c.coefficient[order][4]
                    f = self._c.coefficient[order][5]
                    last_order = order
                y[index] = a + b * np.cos(phi[index])
                h[index] = c * np.cos(2 * phi[index]) + d * np.cos(3 * phi[index]) + e * np.cos(4 * phi[index]) + f * np.cos(5 * phi[index])
#             print("更新后：y[index] + h[index] : {}".format(y[index] + h[index]))
                out_img[j][index] = np.round(img[j][index] + y[index] + h[index])
    
        return out_img

def fringe_wrapper(cfg, method):
    assert method in ["fringe","gamma","harmonic","harmonics_have_noise","seg"]
    d = {"fringe":Fringes,"gamma":Fringes_Gamma,"harmonic":Fringes_Harmonics,"harmonics_have_noise":Fringes_Harmonics_have_noise,"seg":Fringes_Seg}
    return d[method](cfg)
    
def projection_fringes_save():
    from config import config
    cfg = config()  

    cfg.hv="h"
    Fringes(cfg).save_images() 
    cfg.hv="v"
    Fringes(cfg).save_images() 
    

class Fringes_Generator():
    """
    A distributed version of fringes.py. That says, the A is NOT a constant value across the whole image.
    """
    def __init__(self, config):
        self._c = config
        self.SIG = fringe_wrapper(self._c, "harmonic") # synthetic image generator

    def update_cfg(self):
        print("xxxx")
        A0 = np.random.choice(np.linspace(50,150,101)) + np.random.randn()#得到50-150之间一个属
        self._c.A = [np.reshape(np.linspace(A0, 150, self._c.pattern_size[1]), (-1,1))]*3 #生成数组
        A_min, A_max = np.min(self._c.A[0]), np.max(self._c.A[0])
        B_max = np.min([A_min, 255-A_max])
        # self._c.B = [(B_max-10)*np.random.rand()+10]*3
        self._c.B = [B_max*(np.random.rand()+1)/2]*3

        # self._c.A = [128]*3
        # self._c.B = [50]*3

        # self._c.C = 2.5 #返回的是标准正态分布的一个值
        self._c.C = 5*np.random.rand() # -10,10 np.random.rand()返回的是标准正态分布的一个值
        self._c.D = 4*np.random.rand()  # -5,5
        self._c.E = 3*np.random.rand()   # -3,3
        self._c.F = 2*np.random.rand()   # -2,2
        self._c.gamma = 0.8*np.random.rand()+1.5 # 0.7,2.3
        self._c.parameter_array = [A0,self._c.B[0],self._c.C,self._c.D,self._c.E,self._c.F]

#         method = np.random.choice(["fringe", "gamma", "harmonic"])
#         self.SIG = fringe_wrapper(self._c, method) # update the synthetic image generator 
#         print(self._c.B[0], self._c.C, self._c.D, self._c.E, self._c.F, self._c.gamma)


    def save_data(self, ind):
        self.update_cfg() # 整个系统重新配置一遍。
        images = self.SIG.generate_all() # 生成图像
        # if ind < 20:
        #     for f, fringes in enumerate(images):
        #         for s, fringe in enumerate(fringes):
        #             p = Path(self._c.pattern_path)/f"{ind:0>3d}{f:0>2d}{s:0>2d}{self._c.hv}.bmp" 
        #             cv2.imwrite(str(p), fringe)
        np.savez(self._c.net_dir+f"train_data{ind:03d}", image1=images[0], image2=images[1], image3=images[2]) #将3个条纹保存在未压缩的.npz格式中
        print(f"{ind}th syn data is saved")
    

if __name__ == '__main__':
    projection_fringes_save()
