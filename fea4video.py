import cv2
import numpy as np
import os
import math
import scipy.io as sio
import argparse
import dlib
import torch
from models import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 单 GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def VGGFea(frms_seg):
    # frms_seg = PreProcess(frms_seg)
    
    model = VGGEmbedding().to(device)
    model.eval()
    with torch.no_grad():
        input_batch = torch.tensor(frms_seg, dtype=torch.float32).to(device)
        fea = model(input_batch)
    return fea


def gamma_correction(image, gamma=1.5):
    # 构建伽马查找表
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # 应用伽马校正
    return cv2.LUT(image, table)

def PreProcess(frms_seg, crop_size=224):
    
    # 初始化 dlib 人脸检测器和关键点检测器
    try:
        # 尝试加载预训练模型
        # detector = dlib.get_frontal_face_detector()
        cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    except:
        print("正在下载人脸检测模型...")
        import urllib.request
    
    img = frms_seg[1,:,:,:]
    (h,w)=img.shape[:2]
    img = gamma_correction(img)
    
    # 人脸检测
    faces = cnn_detector(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1)
    
    # 如果没有检测到人脸，返回 None
    if len(faces) == 0:
        img = frms_seg[-1,:,:,:]
        img = gamma_correction(img)
        faces = cnn_detector(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1)
        if len(faces)==0:
            print("警告: 未检测到人脸")
            return None
    
    # 选择最大的人脸 (根据边界框面积)
    main_face = max(faces, key=lambda r: (r.rect.right()-r.rect.left()) * (r.rect.bottom()-r.rect.top()))
    
    # 获取人脸关键点
    landmarks = predictor(img, main_face.rect)
    
    # 计算人脸中心点 (使用鼻子位置作为参考)
    nose_bridge = np.array([(landmarks.part(27).x, landmarks.part(27).y)])
    
    # 计算人脸区域大小 (基于关键点)
    all_points = np.array([(p.x, p.y) for p in landmarks.parts()])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    # 计算人脸宽度和高度
    face_width = x_max - x_min
    face_height = y_max - y_min
    
    center_x = (x_min+x_max)//2
    center_y = (y_min+y_max)//2
    
    return frms_seg[:, center_y-crop_size//2:center_y+crop_size//2, center_x-crop_size//2:center_x+crop_size//2,:]
    
    

def videoframes(vf):
    camera = cv2.VideoCapture(vf)
    camera.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    
    frame_skip = 3
    frame_count = 0
    frms = None
    while camera.isOpened():
        sucess, video_frame = camera.read()
        if sucess is False:
            camera.release()
        else:
            # frm_gray = cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY)
            if frame_count % frame_skip == 0:
                if frms is None:
                    frms = np.array([video_frame])
                else:
                    frms = np.concatenate((frms, [video_frame]))
            frame_count = frame_count+1
    return frms

def Fea4Video(vf):
    fps = 10

    frms = videoframes(vf)

    frms_cnts = frms.shape[0]
    segs = frms_cnts//fps
    lbps = None
    
    frms = PreProcess(frms)
    for i in range(segs):
        frms_seg = frms[i*fps:(i+1)*fps,:,:,:]
        lbp = VGGFea(frms_seg)
        lbp = lbp.reshape(1,-1).detach().numpy()
        if lbps is None:
            lbps = lbp
        else:
            lbps = np.vstack((lbps, lbp))
    return lbps
    
def Fea4Subj(rootpath):
    vids = []
    lbps_all = None
    for i in range(7):
        vf = rootpath + str(i) + '.mp4'
        print(vf)
        lbps = Fea4Video(vf)
        if lbps_all is None:
            lbps_all = lbps
        else:
            lbps_all = np.vstack((lbps_all, lbps))
        
        vids = np.append(vids, np.ones(lbps.shape[0], np.int32)*i)

    sio.savemat(rootpath+'videoFea_.mat', {'lbps_all':lbps_all, 'vids':vids})
    

        
def Fea4AllSubjs_(rootpath, subs):
    for i in subs:
        print(f'=========Subject: {i}==========')
        Fea4Subj(rootpath+ str(i)+ '/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--subj_b', type=int, default=1, help='begin subject index')
    #parser.add_argument('--subj_e', type=int, default=35, help='end subject index')
    #args = parser.parse_args()
    #print(args.subj_b, args.subj_e)
    subs =[1,2,3,4,5,8,9,16,17,19,20,21,22,23,24,25,30,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,67,68,69,70,71,72,74,75,76,77,79,80,81,82,83,84,85,86]
    
    Fea4AllSubjs_('D:/人格数据集/data/Personality Data/Aligned Data/', subs)#pls change the aligned data path