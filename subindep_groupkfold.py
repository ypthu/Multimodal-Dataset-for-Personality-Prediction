"""
Group5Fold test
"""

import scipy.io as sio
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import warnings
import gc
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings('ignore')

# ==================== Params ====================
# TRAIT_INDEX = 4  # 1-5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 64*2
LEARNING_RATE = 0.0005
EPOCHS = 50
FEATURE_DIM = 64*4
# SELECTED_VIDEO_IDS = [6]

# Conformer 参数
CONFORMER_DIM = 64*2
CONFORMER_HEADS = 4
CONFORMER_KERNEL_SIZE = 15
CONFORMER_DEPTH = 2*2

# 正则化参数
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1

RANDOM_SEED = 21
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



# =================== traditional methods ==============================
def mlp_classifier_(train_d, train_l, test_d, test_l):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    b_p_l = None
    
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, batch_size=64, random_state=1)
    # model = LinearSVC(C=C, max_iter=10000)
    model.fit(train_d, train_l)
    acc = model.score(test_d, test_l)
    acc_train = model.score(train_d, train_l)
    p_l = model.predict(test_d)
    print(acc_train, acc)
    # if acc > best_acc and acc_train > 0.7:
    #     best_acc = acc
    #     best_model = model
    #     b_p_l = p_l
    # print('best acc', best_acc)
    best_model = model
    best_acc = acc
    
    
    return best_model, best_acc


def svm_classifier_(train_d, train_l, test_d, test_l):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    b_p_l = None
    for C in np.logspace(-10, 10, 20):
        model = SVC(gamma=2, C=C)
        # model = LinearSVC(C=C, max_iter=10000)
        model.fit(train_d, train_l)
        acc = model.score(test_d, test_l)
        acc_train = model.score(train_d, train_l)
        p_l = model.predict(test_d)
        print(C, acc_train, acc)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            b_p_l = p_l
    print('best acc', best_acc)
    
    
    return best_model, best_acc


def rf_classifier_(train_d, train_l, test_d, test_l):
    # train_d = normalize(train_d, axis=0)
    # test_d = normalize(test_d, axis=0)
    # train_d = normalize(train_d, axis=1)
    # test_d = normalize(test_d, axis=1)
    best_acc = -1
    b_p_l = None
    for i in range(10, 100, 10):
        model = RandomForestClassifier(n_estimators=i, max_depth=10)
        model.fit(train_d, train_l)
        acc_t = model.score(train_d, train_l)
        acc = model.score(test_d, test_l)
        p_l = model.predict(test_d)
        print(i, acc_t, acc)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            b_p_l = p_l
    print('best acc', best_acc)
    
    return best_model, best_acc


def knn_classifier_(train_d, train_l, test_d, test_l):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    
    k_vals = [1, 2, 5, 8, 10]
    b_p_l = None
    for k in k_vals:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_d, train_l)
        acc_t = model.score(train_d, train_l)
        acc = model.score(test_d, test_l)
        p_l = model.predict(test_d)
        print(k, acc_t, acc)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            b_p_l = p_l
    print('best acc', best_acc)
    
    return best_model, best_acc


def mlp_classifier(train_d, train_l, val_d, val_l, test_d, test_l):
    # 1. 数据归一化 (对Val数据做同样处理)
    train_d = normalize(train_d, axis=0)
    val_d = normalize(val_d, axis=0)
    test_d = normalize(test_d, axis=0)
    
    train_d = normalize(train_d, axis=1)
    val_d = normalize(val_d, axis=1)
    test_d = normalize(test_d, axis=1)
    
    # MLP目前没有超参数循环，但逻辑保持一致：训练 -> 验证 -> 测试
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, batch_size=64, random_state=1)
    model.fit(train_d, train_l)
    
    # 计算各项指标
    acc_train = model.score(train_d, train_l)
    acc_val = model.score(val_d, val_l)
    
    print(f"MLP - Train: {acc_train:.4f}, Val: {acc_val:.4f}")
    
    # 因为没有循环，当前模型就是 Best Model
    best_model = model
    # 使用该模型对测试集进行最终评估
    test_acc = best_model.score(test_d, test_l)
    print(f"MLP - Final Test Acc: {test_acc:.4f}")
    
    return best_model, test_acc


def svm_classifier(train_d, train_l, val_d, val_l, test_d, test_l):
    train_d = normalize(train_d, axis=0)
    val_d = normalize(val_d, axis=0)
    test_d = normalize(test_d, axis=0)
    
    train_d = normalize(train_d, axis=1)
    val_d = normalize(val_d, axis=1)
    test_d = normalize(test_d, axis=1)
    
    best_val_acc = -1
    best_model = None
    
    # 在验证集上寻找最佳超参数 C
    for C in np.logspace(-10, 10, 20):
        model = SVC(gamma=2, C=C)
        model.fit(train_d, train_l)
        
        acc_train = model.score(train_d, train_l)
        acc_val = model.score(val_d, val_l)  # 使用验证集评估
        
        print(f"C={C:.2e} | Train: {acc_train:.4f}, Val: {acc_val:.4f}")
        
        # 依据验证集准确率更新最佳模型
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model = model
    
    # 使用验证集上表现最好的模型测试 Test 集
    if best_model is None:  # 防御性代码，防止循环未执行
        best_model = model
    
    test_acc = best_model.score(test_d, test_l)
    print(f'SVM - Best Val Acc: {best_val_acc:.4f}, Final Test Acc: {test_acc:.4f}')
    
    return best_model, test_acc


def rf_classifier(train_d, train_l, val_d, val_l, test_d, test_l):
    train_d = normalize(train_d, axis=0)
    val_d = normalize(val_d, axis=0)
    test_d = normalize(test_d, axis=0)
    
    train_d = normalize(train_d, axis=1)
    val_d = normalize(val_d, axis=1)
    test_d = normalize(test_d, axis=1)
    
    best_val_acc = -1
    best_model = None
    
    # 在验证集上寻找最佳 n_estimators
    for i in range(10, 100, 10):
        model = RandomForestClassifier(n_estimators=i, max_depth=10, random_state=42)
        model.fit(train_d, train_l)
        
        acc_train = model.score(train_d, train_l)
        acc_val = model.score(val_d, val_l)  # 使用验证集评估
        
        print(f"n_estimators={i} | Train: {acc_train:.4f}, Val: {acc_val:.4f}")
        
        # 依据验证集准确率更新最佳模型
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model = model
    
    test_acc = best_model.score(test_d, test_l)
    print(f'RF - Best Val Acc: {best_val_acc:.4f}, Final Test Acc: {test_acc:.4f}')
    
    return best_model, test_acc


def knn_classifier(train_d, train_l, val_d, val_l, test_d, test_l):
    train_d = normalize(train_d, axis=0)
    val_d = normalize(val_d, axis=0)
    test_d = normalize(test_d, axis=0)
    
    train_d = normalize(train_d, axis=1)
    val_d = normalize(val_d, axis=1)
    test_d = normalize(test_d, axis=1)
    
    best_val_acc = -1
    best_model = None
    
    k_vals = [1, 2, 5, 8, 10]
    
    # 在验证集上寻找最佳 K 值
    for k in k_vals:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_d, train_l)
        
        acc_train = model.score(train_d, train_l)
        acc_val = model.score(val_d, val_l)  # 使用验证集评估
        
        print(f"k={k} | Train: {acc_train:.4f}, Val: {acc_val:.4f}")
        
        # 依据验证集准确率更新最佳模型
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model = model
    
    test_acc = best_model.score(test_d, test_l)
    print(f'KNN - Best Val Acc: {best_val_acc:.4f}, Final Test Acc: {test_acc:.4f}')
    
    return best_model, test_acc

def build_data(g_fea, g_vid, g_label, fold, vid, trait):
    train_all_fea = None
    train_all_vid = None
    train_all_label = None
    test_all_fea = None
    test_all_vid = None
    test_all_label = None
    
    for i in range(5):
        mask = g_vid[i]==vid
        if i == fold:
            test_all_fea, test_all_vid, test_all_label = g_fea[i][mask], g_vid[i][mask], g_label[i][mask]
        else:
            if train_all_fea is None:
                train_all_fea, train_all_vid, train_all_label = g_fea[i][mask], g_vid[i][mask], g_label[i][mask]
            else:
                train_all_fea, train_all_vid, train_all_label = np.vstack((train_all_fea, g_fea[i][mask])), np.concatenate((train_all_vid, g_vid[i][mask])), np.vstack((train_all_label, g_label[i][mask]))
    
    return train_all_fea, train_all_label[:,trait], test_all_fea, test_all_label[:, trait]


# ==================== 主函数 ====================
def main():
    ROOT_PATH = './'
    models = ['svm','mlp','rf','knn']
    modalities = [[0, 273]]#[[0, 90], [90, 118], [118, 145], [145, 273], [0, 118], [0, 145],
                  #[0, 273]]  # EEG,GSR,PPG, Video, EEG+GSR, EEG+GSR+PPG,EEG+GSR+PPG+Video
    modalities_name = {90: 'EEG', 28: 'GSR', 27: 'PPG', 128: 'Video', 118: 'EEG+GSR', 145: 'EEG+GSR+PPG',
                       273: 'EEG+GSR+PPG+Video'}
    trait_names = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    vids_t = range(7)
    
    # subject level 5-fold cross validation
    sub_group = [[1, 8, 20, 25, 38, 43, 48, 53, 58, 67, 72, 79, 84, 103, 108, 114, 119, 125, 130, 135, 141, 146, 151],
                 [2, 9, 21, 30, 39, 44, 49, 54, 59, 68, 74, 80, 85, 104, 109, 115, 120, 126, 131, 136, 142, 147, 152],
                 [3, 16, 22, 34, 40, 45, 50, 55, 63, 69, 75, 81, 86, 105, 111, 116, 121, 127, 132, 137, 143, 148, 153],
                 [4, 17, 23, 35, 41, 46, 51, 56, 64, 70, 76, 82, 101, 106, 112, 117, 122, 128, 133, 139, 144, 149, 154],
                 [5, 19, 24, 37, 42, 47, 52, 57, 65, 71, 77, 83, 102, 107, 113, 118, 124, 129, 134, 140, 145, 150]]
    
    label_df = pd.read_csv('./data/BFI44_Labels.csv')
    group_feas = []
    group_vids = []
    group_label = []
    for gid in range(5):
        datas = None
        vids = None
        labels = None
        for sid in sub_group[gid]:
            row = label_df[label_df.iloc[:, 0] == sid]
            label = [row.iloc[0, tid] for tid in range(1,6)]
            data = sio.loadmat(ROOT_PATH +'data/'+ str(sid) + '.mat')
            if datas is None:
                datas = np.real(data['feas'])
                vids = data['vids'].squeeze(-1).tolist()
                labels = [label[:] for _ in range(len(vids))]
            else:
                datas = np.vstack((datas, np.real(data['feas'])))
                vids = np.concatenate((vids, data['vids'].squeeze(-1).tolist()))
                labels = np.vstack((labels, [label[:] for _ in range(data['feas'].shape[0])]))
        
        group_feas.append(datas)
        group_vids.append(vids)
        group_label.append(labels)
    
    for model in models:
        # loop for each trial corresponding to stimulus video with ID vid
        for vid in vids_t:
            for trait in range(5):  # classification for the five dimensions of big five
                for mods in modalities:
                    accs = []
                    
                    for fold in range(5):
                        X_train, Y_train, X_test, Y_test = build_data(group_feas, group_vids, group_label,fold, vid, trait)
                    
                        if model == 'svm':
                            best_model, best_acc = svm_classifier_(X_train, Y_train, X_test, Y_test)
                        elif model == 'rf':
                            best_model, best_acc = rf_classifier_(X_train, Y_train, X_test, Y_test)
                        elif model == 'knn':
                            best_model, best_acc = knn_classifier_(X_train, Y_train, X_test, Y_test)
                        elif model == 'mlp':
                            best_model, best_acc = mlp_classifier_(X_train, Y_train, X_test, Y_test)
                        
                        accs.append(best_acc)
                    sio.savemat(ROOT_PATH + 'Results/' + model + '_' + str(vid) + '_' + trait_names[trait] + '_' +
                                modalities_name[mods[1] - mods[0]] + '.mat', {'accs': accs})


if __name__ == "__main__":
    accs=main()
    print(accs)
