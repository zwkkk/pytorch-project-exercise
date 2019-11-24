# -*- coding: utf-8 -*
import numpy
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
import torch

from path import DATA_PATH

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
#device = torch.device('cpu')

class Processor(Base):
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        path = path.replace('\\','/')
        image = Image.open(path).convert('L')
        image = image.crop((32, 32, 223, 223))
        image = image.resize((224, 224))
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = x_data.reshape([224, 224, 1])
        x_data = numpy.multiply(x_data, 1.0 / 255.0)  ## scale to [0,1] from [0,255]
        x_data = numpy.transpose(x_data, (2, 0, 1))  ## reshape
        return x_data

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return numpy.argmax(data)