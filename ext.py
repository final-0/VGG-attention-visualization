import math
import os
import torch
from torch import autograd as autograd
from collections import OrderedDict
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.models import vgg19
import math
import numpy as np
import cv2
import glob
from torchsummary import summary
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import random
from scipy.ndimage.filters import gaussian_filter

class FeatureExtractor37(nn.Module):
    def __init__(self):
        super(FeatureExtractor37, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:37])
    def forward(self, img):
        return self.feature_extractor(img)

def sigmoid37(x):
    return 1/(1+np.exp(-x*100+0.5))


def png_tensor(png):
    trans = T.Compose([T.ToTensor()])
    png = trans(png)
    return png

def get_QP(QP_file,frame,index):
    qp_data = QP_file[0]
    qp = natsorted(glob.glob(qp_data+'/*.png'))
    qp_name = qp[frame]
    print(qp_name)
    img_QP = Image.open(qp_name)
    img_QP = png_tensor(img_QP)
    img_QP = img_QP.unsqueeze(0)
    return img_QP

Test_file = natsorted(glob.glob('BBDrill_half_raw'))
FeatureExtractor37 = FeatureExtractor37().cuda()

for epoch in range(1):
    print(epoch)
    test_frame_nums = 500
    test_video_nums = 1
    
    for test_index in tqdm(range(test_video_nums)):
        for tt in range(0, test_frame_nums):

            input_image = get_QP(Test_file,tt,index=test_index)
            input_image1 = input_image.cuda()
            sizeX = input_image.size()[3]
            sizeY = input_image.size()[2]
            with torch.no_grad():
                output_image37 = FeatureExtractor37(input_image1)

            num_c37 = output_image37.size()[1]
            size_y37 = output_image37.size()[2]
            size_x37 = output_image37.size()[3]
            importance37 = np.zeros((size_y37,size_x37))

            for i in range(num_c37):
                feature37 = output_image37[:,i,:,:]
                feature37 = feature37.detach().cpu().numpy()
                feature37 = np.squeeze(feature37)
                feature37 = sigmoid37(feature37)
                f_sum37 = np.sum(feature37)
                weight37 = 1-(f_sum37/size_y37/size_x37)
                importance37 += (feature37*weight37)**2/num_c37
            i_map37 = np.squeeze(importance37)
            i_max37 = np.max(i_map37)
            i_min37 = np.min(i_map37)
            i_map37 = (i_map37-i_min37)/(i_max37-i_min37)
            i_map37 = cv2.resize(i_map37, dsize=(sizeX, sizeY), interpolation=cv2.INTER_LINEAR)

            input_image = input_image.detach().cpu().numpy()
            
            #i_map37 = i_map37.clip(0,0.5)
            #i_map37 = (i_map37)/0.5
            i_map37 = np.where(i_map37>=0.3, 1.0,0.0)
            i_map37 = i_map37.clip(0,1.0)
            i_map37 = gaussian_filter(i_map37,sigma=11)
            i_map37 = i_map37*input_image
    
            i_max37 = np.max(i_map37)
            i_min37 = np.min(i_map37)
            i_map37 = (i_map37-i_min37)*255/(i_max37-i_min37)
            #i_map37 = i_map37*255
            i_map37 = i_map37.clip(0,255)
            i_map37 = np.squeeze(i_map37)
            i_map37 = i_map37.transpose(1,2,0)
            i_map37 = i_map37[:, :, [2, 1, 0]]
            #finally_image = cv2.cvtColor(finally_image, cv2.COLOR_BGR2RGB)
           
            cv2.imwrite('./feature/image%03d.png'%(tt+1),i_map37.astype(np.uint8))
            
        
