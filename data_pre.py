# -*- coding: utf-8 -*-
#生成图片路径txt文件
import os
fname='temp.txt'
root='/home/a504/PycharmProjects/movedata/fm/sz_chb01_16_1_1'
with open("temp.txt", "w") as f:
    im = os.listdir(root)
    for im in os.listdir(root):
        imroot=root+'/'+im+' 0'
        f.write(str(imroot)+ '\n' )