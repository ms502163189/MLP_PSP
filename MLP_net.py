import numpy as np
from torch import nn
import os
import torch,gc
from config import config

# 定义训练的设备
device = torch.device("cuda") #使用gpu进行训练
gc.collect()
torch.cuda.empty_cache()#清楚cuda缓存

class NeuralNetwork(nn.Module): #构建CNN神经网络
    def __init__(self,config):
        super(NeuralNetwork, self).__init__()
        
        self._c = config
        self.MLP = nn.Sequential(
            nn.Conv2d(9,  64, 1, stride=1),  #四个卷积层
            nn.ReLU(),
            nn.Conv2d(64, 256, 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, 1, stride=1),
        )
        

        A = [[1, np.cos(0), np.sin(0)],                
             [1, np.cos(6*np.pi/7), np.sin(6*np.pi/7)],
             [1, np.cos(8*np.pi/7), np.sin(8*np.pi/7)]]
        B = [[1, np.cos(2*np.pi/7), np.sin(2*np.pi/7)],
             [1, np.cos(4*np.pi/7), np.sin(4*np.pi/7)],
             [1, np.cos(10*np.pi/7), np.sin(10*np.pi/7)],
             [1, np.cos(12*np.pi/7), np.sin(12*np.pi/7)]]
        A, B = np.array(A), np.array(B) #列表转化成数组
        self.C = torch.Tensor(np.matmul(B,np.linalg.inv(A)).transpose()).float()#B乘以A的转置矩阵就是I4到I7的前面的系数
        self.C = self.C.to(device)
        print("self.C : ",self.C)
        D = [[1, np.cos(0), np.sin(0)],
            [1, np.cos(2*np.pi/4), np.sin(2*np.pi/4)],
            [1, np.cos(4*np.pi/4), np.sin(4*np.pi/4)]]
        E = [[1, np.cos(6*np.pi/4), np.sin(6*np.pi/4)]]
        self.F = torch.Tensor(np.matmul(E, np.linalg.pinv(D)).transpose()).float()#B乘以A的转置矩阵就是I4到I7的前面的系数
        self.F = self.F.to(device)
        self.model_save_dir = self._c.model_dir #定义保存模型的文件夹
        self.p = os.path.join(self.model_save_dir,"best_model")
        self.last_loss = 9999999999
        self.last_epoch = 0
        
        #对数组A求逆矩阵，与B矩阵相乘
    
    def forward(self, images):
        # linear generator for the 4 images
        images = images.to(device)
        i7s = images.permute(0, 2, 3,1)            #将tensor的维度换位
        # i7s = i7s.to(device)
        i7s,_ = torch.split(i7s,(3,6),dim=-1)      #切分，得到的就是I1 I7 I8
        # mids,another = torch.split(_,(3,3),dim = -1) #切分，得到4步相移的前三步
        # print("self.C.shape",self.C.shape)
        # print("i7s.shape",i7s.shape)

        linearPart = torch.matmul(i7s, self.C+0.0) #Y矩阵乘法 50*(1950*3)矩阵乘以3*4矩阵
        # linearPart1 = torch.matmul(mids, self.F+0.0)   #矩阵乘法 50*(1950*3)矩阵乘以3*1矩阵
        linearPart = linearPart.to(device)
        # linearPart1 = linearPart1.to(device)
        linearPart = linearPart.permute(0,3,1,2)   #维度换位
        # linearPart1 = linearPart1.permute(0,3,1,2)   #维度换位
        # linearPart2 = torch.cat([linearPart, linearPart1], 1)
        
        # The nonliear part is estimated with MLP function 
        images = images.to(device)
        images = images/255
        images = images.to(device)
        res = self.MLP(images)
        return linearPart + res
    
    def save(self,optimizer,epoch,loss,batch): #TODO
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        m = os.path.join(self.model_save_dir,"model"+str(epoch))

        state = {'model':self.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        # 保存模型
#         if epoch % 100 == 0:
#             torch.save(state, m)
#             print("模型{}保存成功".format(str(epoch)))
        
        if (epoch == 0 or self.last_loss == 10000):
            self.last_loss = loss
            self.last_epoch = epoch
            torch.save(state, self.p)
            # print("lat_loss {},best_model保存成功,它是epoch{},loss为{},batch为{}".format(self.last_loss,epoch,loss,batch))
        else:
            if (loss <= self.last_loss): #用loss判断最优模型
                self.last_loss = loss
                self.last_epoch = epoch
                torch.save(state, self.p)
                # print("lat_loss {},best_model保存成功,它是epoch{},loss为{},batch为{}".format(self.last_loss,epoch,loss,batch))
            else:
                pass
                # print("best_model未保存，最优模型仍为epoch{}".format(self.last_epoch,self.last_loss))
        
    def load(self,optimizer): #TODO

        # 如果有保存的模型，则加载模型，并在其基础上继续训练
        if os.path.exists(self.p):
            checkpoint = torch.load(self.p)
            self.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载best_model成功！其为epoch{}'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        return start_epoch