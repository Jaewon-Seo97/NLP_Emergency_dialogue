from codecs import ignore_errors
import tensorflow as tf
import pandas as pd
import json
import os, glob, sys
from collections import OrderedDict, Counter
from konlpy.tag import Okt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score,roc_curve,auc, roc_auc_score, multilabel_confusion_matrix, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import joblib

import datetime

import argparse


def removing_stopword(X_train, y_train, f_train, name_dict, stopwords, n_addSW = 30, acceptList = ['Verb', 'Noun', 'Adjective', 'Determiner']):
    
    X_train_token = pd.DataFrame(columns=['class', 'morphs', 'f_name'])
    for i, x in enumerate(X_train):
        X_train_pos = Okt().pos(x, norm=True, stem=True)
        X_train_pos = list(set([item[0] for n, item in enumerate(X_train_pos) if (item[1] in acceptList) and (item[0] not in stopwords)]))
        sub_token = pd.DataFrame({'class': y_train.values[i],
                                             'morphs': [X_train_pos],
                                              'f_name': f_train.values[i]})
        X_train_token = pd.concat([X_train_token, sub_token], ignore_index=True)
                                       
    overlaps = []
    for w in list(name_dict.keys()):
        # print(f'{w} {name_dict[w]}')
        word_count = Counter(np.hstack(X_train_token[X_train_token['class'] == w]['morphs']))

        # add stop word
        for k in range(n_addSW):
            overlaps.append(word_count.most_common(n_addSW)[k][0])

    overCount = Counter(np.hstack(overlaps))
    # print(overCount)

    over_idx = list(overCount.values())
    over_key = list(overCount.keys())
    
    for oid, n_over in enumerate(over_idx):
        if n_over == len(list(name_dict.keys())):
            stopwords.append(over_key[oid])
    # print(stopwords)    
    
    return X_train_token, stopwords

def pos_select(X_train, y_train, f_train, stopwords, acceptList = ['Verb', 'Noun', 'Adjective', 'Determiner']):
    X_train_token = pd.DataFrame(columns=['file_name', 'class', 'morphs'])
    for i, x in enumerate(X_train):

        X_train_pos = Okt().pos(x, norm=True, stem=True)
        X_train_pos = list(set([item[0] for n, item in enumerate(X_train_pos) if (item[1] in acceptList) and (item[0] not in stopwords)]))
        sub_token = pd.DataFrame({'class': y_train.values[i],
                                         'morphs': [X_train_pos],
                                             'file_name': f_train.values[i]})
        X_train_token = pd.concat([X_train_token, sub_token], ignore_index=True)
        
    return X_train_token

def checking_keyword_freq(path_dataDF, name_dict, X_train_token, n_top = 10, n_addSW = 30):
    os.makedirs(os.path.join(path_dataDF, 'keywords_img'), exist_ok=True)
    for w in range(len(name_dict)):
        print(f'{w} {name_dict[w]}')
        word_count = Counter(np.hstack(X_train_token[X_train_token['class'] == w]['morphs']))
        top_words = word_count.most_common(n_top)

        keywords = [x for x, y in top_words]
        hist = [y for x, y in top_words]
        print(keywords)

        plt.figure(figsize= (8,10))
        plt.rc('font', family='NanumBarunGothic')
        plt.barh(range(len(keywords)), hist)
        plt.yticks(range(len(keywords)), keywords, fontsize=20)
        plt.xticks(fontsize=20)
        plt.savefig(os.path.join(path_dataDF, 'keywords_img', f'{name_dict[w]} keyword_freq_{n_addSW}.png'))
        # plt.show() 

def _making_token(X_token):
    X_train = np.asarray(X_token['morphs'])

    corp_Xtrain = []
    for x in X_train:
        init_txt = ""
        for xt in x:
            init_txt += f' {xt}'
        corp_Xtrain.append(init_txt[1:])

    return corp_Xtrain
class _get_tfidf_ml:
    def __init__(self, corp_Xtrain, y_train, savepath, name_dict, max_features=None,ngram_range=(1,1), date = datetime.date.today()):
        self.vectorizer = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=1)
        self.vectorizer.fit_transform(corp_Xtrain)    
        self.corp_Xtrain = corp_Xtrain
        self.y_train = y_train
        self.savepath = savepath
        self.name_dict = name_dict
        self.date = date

    def ml_class(self, clf, corp_Xtrain, y_train, clf_name):
        print(f'Training {clf_name}')
        model= Pipeline([("vectorizer", self.vectorizer), ("classifier", clf)])
        start_time = datetime.datetime.now()
        model.fit(corp_Xtrain, y_train)
        end_time = datetime.datetime.now()

        training_time_tfidf = (end_time - start_time).total_seconds()
        print('Training time: {:.1f}s'.format(training_time_tfidf)) 

        return model

    def ml_predict(self, model, corp_Xtest, y_test, f_test, clf_name):

        test_y = tf.keras.utils.to_categorical(y_test)
        test_y = test_y[:,list(self.name_dict.keys())]
        
        print(f'Prediction: {clf_name}')
        predicted_train_tfidf = model.predict(self.corp_Xtrain)
        accuracy_train_tfidf = accuracy_score(self.y_train, predicted_train_tfidf)
        print('Accuracy Training data: {:.1%}'.format(accuracy_train_tfidf))

        predicted_test_tfidf = model.predict(corp_Xtest)
        accuracy_test_tfidf = accuracy_score(y_test, predicted_test_tfidf)
        print('Accuracy Test data: {:.1%}'.format(accuracy_test_tfidf))

        proba = model.predict_proba(corp_Xtest)

        # Saving result values

        mcof_mat = multilabel_confusion_matrix(y_test, predicted_test_tfidf)
        f1_each = f1_score(y_test, predicted_test_tfidf, average=None)
        mcmList = pd.DataFrame(columns=['class', 'f1-score', 'AUC', 'TP', 'FP', 'TN', 'FN'])
        print(list(self.name_dict.values()))
        print(len(mcof_mat))
        aucs = []
        for c, cm in enumerate(mcof_mat):

            name_cls = f'{list(self.name_dict.values())[c]}'
            each_auc = roc_auc_score(test_y[:, c], proba[:, c], multi_class="ovr")
            print(f'confusion matrix: {name_cls}')
            print(cm)
            print(f'AUC: {each_auc}')
            cat_cm = pd.DataFrame({'class': name_cls,
                                    'f1-score': f'{f1_each[c]:0.2f}',
                                    'AUC': f'{each_auc:0.2f}',
                                    'TN': cm[0][0],
                                    'FP': cm[0][1],
                                    'FN': cm[1][0],
                                    'TP': cm[1][1]}, index=[name_cls])
            mcmList = pd.concat([mcmList, cat_cm], axis = 0)
            aucs.append(each_auc)
        mcmList.to_excel(os.path.join(self.savepath, f'{clf_name}_Multi-ConfusionMatrix_{self.date}.xlsx'))
        cof_mat = confusion_matrix(y_test, predicted_test_tfidf)
        cof_mat = pd.DataFrame(cof_mat, columns=list(self.name_dict.values()), index = list(self.name_dict.values()))
        cof_mat = pd.concat([cof_mat, pd.DataFrame({'f1_score_macro':f1_score(y_test, predicted_test_tfidf, average='macro'),
                                                        'AUC_macro':f'{sum(aucs)/len(aucs):0.3f}',
                                                        'accurcy': f'{accuracy_test_tfidf:0.3f}',
                                                        }, index = ['Total'])])
        cof_mat.to_excel(os.path.join(self.savepath, f'{clf_name}_ConfusionMatrix_{self.date}.xlsx'))
        
        
        probclassList = [f'prob_{self.name_dict[name]}' for name in list(self.name_dict.keys())]
        probList = pd.DataFrame(proba, columns=probclassList)

        resultList = pd.DataFrame(columns=['file_name', 'GT', 'Prediction'])
        fail_idx = []
        i = 0
        for gt, yp in zip(np.array(y_test), predicted_test_tfidf):
            gt_name = self.name_dict[gt]
            pr_name = self.name_dict[yp]
            sub_df = pd.DataFrame({'file_name': f'{np.array(f_test)[i]}', 
                               'GT': gt_name, 
                               'Prediction':pr_name,
                               }, index=[i])
            resultList = pd.concat([resultList, sub_df])
            
            if gt != yp:
                fail_idx.append(i)
               
                
            i+=1
        resultList = pd.concat([resultList, probList], axis=1)
        failList = resultList.loc[fail_idx]
        failList.to_excel(os.path.join(self.savepath, f'{clf_name}_FailList_{self.date}.xlsx'))
        resultList.to_excel(os.path.join(self.savepath, f'{clf_name}_ResultList_{self.date}.xlsx'))
        return proba, probList, resultList
    
    def drawing_ROCcurve(self, y_test, test_result, savepath, clf_name, name_dict):
        test_y = tf.keras.utils.to_categorical(y_test)
        test_y = test_y[:,list(name_dict.keys())]
        n_classes = test_y.shape[-1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_y[:, i], test_result[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), test_result.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        # First aggregate all false positive rates
        lw = 2
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(10,10))
        plt.rc('font', family='NanumBarunGothic')
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average (auc = {0:0.2f})".format(roc_auc["micro"]),
            color="maroon",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average (auc = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = ["b", "r", "g", "c", "m"]
        i = 0
        for nc, color in zip(list(name_dict.keys()), colors[:n_classes]):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="class {0} (auc = {1:0.2f})".format(self.name_dict[nc], roc_auc[i]),
            )
            i+=1

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.rc('font', family='NanumBarunGothic')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.title("Receiver operating characteristic curves", fontsize=20)
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(f'{savepath}/ROCcurve_{clf_name}_{n_classes}class_{datetime.date.today()}.png')
        
        return fpr["micro"], tpr["micro"], roc_auc["micro"], fpr["macro"], tpr["macro"], roc_auc["macro"]

    
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_class',     default='1,2')
    parser.add_argument('--gpu',                default='7')
    parser.add_argument('--Using_stopword',     action='store_true')
    parser.add_argument('--stopword_file',      default='2022-08-28')

    return parser.parse_args(args)

def main(args):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    StopDetail = args.stopword_file
    path_dataDF = os.path.join(os.path.abspath('.')[:os.path.abspath('.').index('NIA_Emergency')], 
                               'NIA_Emergency', 'data')
                              
    date = datetime.date.today()
    path_save = os.path.join(path_dataDF,'Result', f'class{args.selected_class}_{date}') 
    os.makedirs(os.path.join(path_save, 'excel_file'), exist_ok=True)
    classList = args.selected_class.split(',')
    classList = [int(cl) for cl in classList]
    ######################### 
    # Setting parameter

    init_name_dict = {0:'예진',
              1:'초진',
              2:'투약 및 검사',
              3:'검사결과 설명',
              4:'퇴실'}

    ## setting process categories
    class_dict = dict()
    name_dict = dict()
    
    for i, cl in enumerate(classList):
        name_dict[cl] = init_name_dict[classList[i]]
        class_dict[init_name_dict[classList[i]]] = cl
    
    acceptList = ['Verb', 'Noun', 'Adjective', 'Determiner']

    n_addSW = 30

    # loading data
    init_df_all = pd.read_excel(os.path.join(path_dataDF, 'excel_file', 'All_text_0823.xlsx'), engine='openpyxl')
    print('\nLoading data...')

    for i, cl in enumerate(classList):
        if i == 0:
            df_all = init_df_all[(init_df_all['class']== cl)]
        else:
            df_all = pd.concat([df_all, init_df_all[(init_df_all['class']== cl)]], axis=0)

    print(f'Class: {classList}')
    df_all.reset_index(drop=True,inplace=True)
    df_all.to_excel(os.path.join(path_save, 'excel_file', f'class{args.selected_class}_text_0823.xlsx'))

    x_data = df_all['context_wEng']
    y_data = df_all['class']
    f_name = df_all['file_name']

    
    X_train, X_test, y_train, y_test, f_train, f_test= train_test_split(x_data, y_data, f_name, test_size = 0.2, random_state = 42, stratify=y_data)
    print(f'Train data shape: {X_train.shape}| Test data shape:{X_test.shape}')    

    ##########################
    # Setting stopwords

    if args.Using_stopword:
        after_stopwords = pd.read_excel(os.path.join(path_dataDF, 'StopWord_List', f'class{args.selected_class}_StopWord_{StopDetail}.xlsx'))['stop_words'].to_list()
        print(f'Stop words: {after_stopwords}')
    else:
        # Making stopwords
        ##  init stopwords
        init_stopwords = ['(())', '네', '네네', '네네네', '예', '아', '응', '그', '음', '흠', '오', '어', '으', '우', 
                        '게', '제', '요', '거','쪼끔', '쪼금', '아니', '근데', '조금', '좀', '쫌', '아유', '아야', 
                        '뭐', '아야야', '저', '저희', '나', '일단은','수', '도', '는', '다', '의', '가', '이', '은', 
                        '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯','되', '면', '더', 
                        '돼요', '해요']
    
        # Making stopword list iteratively
        n_addSW = 30
        i=0
        while True:
            print(f'Removing stopword iter: {i}')

            X_train_token, after_stopwords = removing_stopword(X_train, y_train, f_train, name_dict, stopwords=init_stopwords.copy(), n_addSW = n_addSW)
            if (len(after_stopwords) == len(init_stopwords)):
                break
            else:
                print(len(after_stopwords), len(init_stopwords))
                init_stopwords = after_stopwords
            i+=1
        
        pd.DataFrame(init_stopwords, columns=['stop_words']).to_excel(os.path.join(path_save, 'excel_file', f'class{args.selected_class}_StopWord_{date}.xlsx'))
        pd.DataFrame(init_stopwords, columns=['stop_words']).to_excel(os.path.join(path_dataDF, 'StopWord_List', f'class{args.selected_class}_StopWord_{date}.xlsx'))


    #################################    
    # removing final stopwords
    
    X_train_token = pos_select(X_train, y_train, f_train, stopwords=after_stopwords, acceptList = acceptList)
    X_test_token = pos_select(X_test, y_test, f_test, stopwords=after_stopwords, acceptList = acceptList)

    # saving token list
    X_train_token.to_excel(os.path.join(path_save, 'excel_file', f'X_train_token_{date}.xlsx'))
    X_test_token.to_excel(os.path.join(path_save, 'excel_file', f'X_test_token_{date}.xlsx'))
    
    # checking keyword of frequency top 10
    # checking_keyword_freq(path_save, name_dict, X_train_token, n_top = 10, n_addSW=n_addSW)

    ########################################
    # making corpus
    corp_Xtrain = _making_token(X_train_token)
    corp_Xtest = _making_token(X_test_token)


    ###################################
    # training machine learning model

    max_features = 15000
    max_ngram = 2
    n_class = len(name_dict)
    date = datetime.date.today()

    path_savemodel = os.path.join(path_save, f'model_{n_class}class_mxfeature{max_features}_{max_ngram}gram_{date}')
    path_result = os.path.join(path_save, f'ROC_curves')

    os.makedirs(path_savemodel, exist_ok= True)
    os.makedirs(path_result, exist_ok= True)

    tfml = _get_tfidf_ml(corp_Xtrain, y_train, savepath = path_savemodel, name_dict = name_dict, max_features=max_features,ngram_range=(1,max_ngram))

    clf_lr = LogisticRegression()
    clf_svm = svm.SVC(decision_function_shape='ovo', probability=True)
    clf_mlp = MLPClassifier()
    clf_xgb = XGBClassifier()
    clf_rf = RandomForestClassifier()
    clf_et = ExtraTreesClassifier()

    model_lr = tfml.ml_class(clf_lr, corp_Xtrain, y_train, 'LR')
    model_svm = tfml.ml_class(clf_svm, corp_Xtrain, y_train, 'SVM')
    model_xgb = tfml.ml_class(clf_xgb, corp_Xtrain, y_train, 'XGB')
    model_mlp = tfml.ml_class(clf_mlp, corp_Xtrain, y_train, 'MLP')
    model_rf = tfml.ml_class(clf_rf, corp_Xtrain, y_train, 'RF')
    model_et = tfml.ml_class(clf_et, corp_Xtrain, y_train, 'ExtraTree')

    nameList = ['LR', 'SVM', 'XGB', 'MLP', 'RF', 'ExtraTree']
    # nameList = ['LR']
    all_fpr = dict()
    all_tpr = dict()
    each_aucs = dict()
    # for model, clf_name in zip([model_lr], nameList):
    for model, clf_name in zip([model_lr, model_svm, model_xgb, model_mlp, model_rf, model_et], nameList):
        joblib.dump(model, os.path.join(path_savemodel, f'{clf_name}_{datetime.date.today()}.pkl'))
        proba, probList, resultList  = tfml.ml_predict(model, corp_Xtest, y_test, f_test, clf_name)
        mic_fpr, mic_tpr, mic_roc_auc, mac_fpr, mac_tpr, mac_roc_auc= tfml.drawing_ROCcurve(y_test, proba, savepath= path_result, clf_name = clf_name, name_dict = name_dict)
        all_fpr[f"{clf_name}_fpr"] = mac_fpr
        all_tpr[f"{clf_name}_tpr"] = mac_tpr
        each_aucs[f"{clf_name}_auc"] = mac_roc_auc


    # Saving ROC curves: macro average
    plt.figure(figsize=(10,10))
    plt.rc('font', family='NanumBarunGothic')
    for clf_name in nameList:
        each_auc = each_aucs[f'{clf_name}_auc']
        plt.plot(
            all_fpr[f"{clf_name}_fpr"],
            all_tpr[f"{clf_name}_tpr"],
            label=f"{clf_name} (auc = {each_auc:0.2f})",
            linestyle="-",
            linewidth=4,
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.rc('font', family='NanumBarunGothic')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.title("ALL ROC curves", fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.savefig(f'{path_result}/ROCcurve_all_class{args.selected_class}_{date}.png')

if __name__=='__main__':
    main(args = None)