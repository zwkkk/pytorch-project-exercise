# -*- coding: utf-8 -*-
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageEnhance
import torch
class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        return x_train, y_train, x_test, y_test


def batch_X_transform(Xs,transforms):

    imgs=[]
    for x in Xs:
        img=Image.fromarray(x)
        img=transforms(img)
        imgs.append(img)
    imgs=torch.stack(imgs,dim=0)
    return imgs


def src(Xs,size):
    tran= transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),

                                        ])
    return batch_X_transform(Xs, tran)