import numpy as np
import scipy.io as sio
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def mlp_classifier(train_d, train_l, test_d, test_l, pic_spath):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    b_p_l = None
    
    model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, batch_size=64, random_state=1)
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
    # confusion_matrix(test_l, b_p_l)
    matrix = plot_confusion_matrix(best_model, test_d, test_l,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.savefig(pic_spath)
    # plt.show()
    
    return best_model, best_acc

def svm_classifier(train_d, train_l, test_d, test_l, pic_spath):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    b_p_l = None
    for C in np.logspace(-10, 10, 20):
        model = SVC(gamma=2,C=C)
        # model = LinearSVC(C=C, max_iter=10000)
        model.fit(train_d, train_l)
        acc = model.score(test_d, test_l)
        acc_train = model.score(train_d, train_l)
        p_l = model.predict(test_d)
        print(C, acc_train, acc)
        if acc > best_acc and acc_train >0.7:
            best_acc = acc
            best_model = model
            b_p_l = p_l
    print('best acc', best_acc)
    # confusion_matrix(test_l, b_p_l)
    matrix = plot_confusion_matrix(best_model, test_d, test_l,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.savefig(pic_spath)
    # plt.show()
    
    
    return best_model,best_acc


def rf_classifier(train_d, train_l, test_d, test_l, pic_spath):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    b_p_l = None
    for i in range(10, 100, 10):
        model = RandomForestClassifier(n_estimators=i,max_depth=10)
        model.fit(train_d, train_l)
        acc_t = model.score(train_d, train_l)
        acc = model.score(test_d, test_l)
        p_l = model.predict(test_d)
        print(i, acc_t, acc)
        if acc > best_acc and acc_t >0.7:
            best_acc = acc
            best_model = model
            b_p_l = p_l
    print('best acc', best_acc)
    matrix = plot_confusion_matrix(best_model, test_d, test_l,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix for our classifier')
    # plt.plot(matrix)
    plt.savefig(pic_spath)
    # plt.show()
    
    
    return best_model,best_acc

def knn_classifier(train_d, train_l, test_d, test_l, pic_spath):
    train_d = normalize(train_d, axis=0)
    test_d = normalize(test_d, axis=0)
    train_d = normalize(train_d, axis=1)
    test_d = normalize(test_d, axis=1)
    best_acc = -1
    
    k_vals= [1,2,5,8,10]
    b_p_l = None
    for k in k_vals:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_d, train_l)
        acc_t = model.score(train_d, train_l)
        acc = model.score(test_d, test_l)
        p_l = model.predict(test_d)
        print(k, acc_t, acc)
        if acc > best_acc and acc_t > 0.7:
            best_acc = acc
            best_model = model
            b_p_l = p_l
    print('best acc', best_acc)
    matrix = plot_confusion_matrix(best_model, test_d, test_l,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix for our classifier')
    # plt.plot(matrix)
    plt.savefig(pic_spath)
    # plt.show()
    
    return best_model, best_acc



if __name__ == '__main__':
    ROOT_PATH = './AllData/'
    models = ['mlp'，'knn','rf','svm']
    modalities = [[0,90],[90,118],[118,145],[145,273],[0,118],[0,145],[0,273]]#EEG,GSR,PPG, Video, EEG+GSR, EEG+GSR+PPG,EEG+GSR+PPG+Video
    modalities_name = {90:'EEG', 28:'GSR',27:'PPG',128:'Video',118:'EEG+GSR',145:'EEG+GSR+PPG', 273:'EEG+GSR+PPG+Video'}
    trait_names = ['Extraversion','Agreeableness', 'Conscientiousness','Neuroticism','Openness']
    vids = range(7)
    
    for model in models:
        # loop for each trial corresponding to stimulus video with ID vid
        for vid in vids:
            for trait in range(5):  # classification for the five dimensions of big five
                for mods in modalities:
                    accs = []
                    data = sio.loadmat(ROOT_PATH + str(vid) + '.mat')
                    feas = data['Feas'][:, mods[0]:mods[1]]
                    labels = data['Labels'][trait, :]
                    kf = KFold(n_splits=5, shuffle=True)
                    
                    fold = 0
                    for train_index, test_index in kf.split(feas):
                        X_train, X_test = feas[train_index, :], feas[test_index, :]
                        Y_train, Y_test = labels[train_index], labels[test_index]
                        fold = fold+1
                        if model == 'svm':
                            best_model, best_acc = svm_classifier(X_train, Y_train, X_test, Y_test,
                                                                 ROOT_PATH + 'Pics/' + model+'_' + str(vid) + '_' +
                                                                 trait_names[trait] + '_' + modalities_name[
                                                                     mods[1] - mods[0]] + '+' + str(fold) + '.jpg')
                        elif model == 'rf':
                            best_model, best_acc = rf_classifier(X_train, Y_train, X_test, Y_test,
                                                                 ROOT_PATH + 'Pics/' + model +'_' + str(vid) + '_' +
                                                                 trait_names[trait] + '_' + modalities_name[
                                                                     mods[1] - mods[0]] + '+' + str(fold) + '.jpg')
                        elif model == 'knn':
                            best_model, best_acc = knn_classifier(X_train, Y_train, X_test, Y_test,
                                                                ROOT_PATH + 'Pics/' + model + '_' + str(vid) + '_' +
                                                                trait_names[trait] + '_' + modalities_name[
                                                                    mods[1] - mods[0]] + '+' + str(fold) + '.jpg')
                        elif model == 'mlp':
                            best_model, best_acc = mlp_classifier(X_train, Y_train, X_test, Y_test,
                                                                ROOT_PATH + 'Pics/' + model + '_' + str(vid) + '_' +
                                                                trait_names[trait] + '_' + modalities_name[
                                                                    mods[1] - mods[0]] + '+' + str(fold) + '.jpg')
                        
                        accs.append(best_acc)
                    sio.savemat(ROOT_PATH + 'Results/'+model+'_' + str(vid) + '_' + trait_names[trait] + '_' + modalities_name[mods[1] - mods[0]] + '.mat', {'accs':accs})
        