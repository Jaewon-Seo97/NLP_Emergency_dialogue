import pandas as pd
import json
import os
import numpy as np
from konlpy.tag import Okt


def preprocessing_data2morph(path_dataDF, class_dict):

    """
    # class_dict = {'01.예진':0,
                '02.초진':1,
                '03.투약및검사':2,
                '04.검사결과설명및퇴실':3}

    # Using text data
    """

    # fold_json = os.path.join('라벨링데이터')
    path_json = os.path.join(path_dataDF, '라벨링데이터')

    classList = list(class_dict.keys())
    df_all = pd.DataFrame(columns=['class', 'file_name', 'context'])
    for label in classList:
        n= 0    ## To checking the number of data per class

        # loading data and getting values(class, file name, context of dialogue)
        fold_ktas = os.listdir(os.path.join(path_json, label))
        
        for c, fold in enumerate(fold_ktas):
            
            each_classjson = os.listdir(os.path.join(path_json, label, fold))
            each_classjson = [name for name in each_classjson if name.endswith('.json')]
            
            for each_json in each_classjson:
                with open(os.path.join(path_json, label, fold, each_json), "r", encoding="UTF-8") as e_json:
                    each_dict = json.load(e_json)

                context= ''
                for e, annot in enumerate(each_dict['annotations']):

                    cntxt = annot['original_form']

                    if e == 0:
                        context += f'{cntxt}'
                    else:
                        context += f' {cntxt}'


                sub_df = pd.DataFrame({
                            'class': class_dict[label], 
                            'file_name':each_json.split('.')[0],
                            'context':[context]
                            })

                df_all = pd.concat([df_all, sub_df])
                n+=1
            
        print(f'n {label}: {n} case')  # Checking the number of data.

    print('Loading context done!')

    return df_all   

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

def _making_token(X_token):
    X_train = np.asarray(X_token['morphs'])

    corp_Xtrain = []
    for x in X_train:
        init_txt = ""
        for xt in x:
            init_txt += f' {xt}'
        corp_Xtrain.append(init_txt[1:])

    return corp_Xtrain

class get_input_BiLSTM():
    def __init__(self, X_token):
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self.X_token = X_token

    def get_vocabSize(self, token_th=2):

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.X_token['morphs'])
        total_cnt = len(tokenizer.word_index)
        rare_cnt = 0 
        total_freq = 0
        rare_freq = 0

        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value

            if(value <token_th):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        print('단어 집합(vocabulary)의 크기 :',total_cnt)
        print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(token_th - 1, rare_cnt))
        print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
        print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

        vocab_size = total_cnt - rare_cnt + 2
        print('단어 집합의 크기 :',vocab_size)

        return vocab_size

    # def pad_token(nested_list):
    #     tokenizer = Tokenizer(oov_token="<OOV>")
    #     tokenizer.fit_on_texts(X_token['morphs'])
    #     X_train = tokenizer.texts_to_sequences(X_token['morphs'])

    #     def below_threshold_len(max_len, ):
    #         count = 0
    #     for sentence in nested_list:
    #         if(len(sentence) <= max_len):
    #             count = count + 1
    #     print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))


def ml_predict(test_prediction, acc_score, y_test, f_test, clf_name, name_dict, path_savemodel, date):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score,roc_curve,auc, roc_auc_score, multilabel_confusion_matrix, confusion_matrix, f1_score

    # test_y = tf.keras.utils.to_categorical(y_test)
    # test_y = y_test[:,list(name_dict.values())]

    # predicted_train_tfidf = model.predict(X_train)
    # predicted_test_tfidf = model.predict(X_test)
    # proba = predicted_test_tfidf

    # test_prediction = predicted_test_tfidf.argmax(axis=1)
    # accuracy_train_tfidf = accuracy_score(y_train, predicted_train_tfidf.argmax(axis=1))
    # accuracy_test_tfidf = accuracy_score(y_test, test_prediction)
    # print('Accuracy Training data: {:.1%}'.format(accuracy_train_tfidf))
    # print('Accuracy Test data: {:.1%}'.format(accuracy_test_tfidf))
    
    # Saving result values

    mcof_mat = multilabel_confusion_matrix(y_test.argmax(axis=1), test_prediction.argmax(axis=1))
    f1_each = f1_score(y_test.argmax(axis=1), test_prediction.argmax(axis=1), average=None)
    mcmList = pd.DataFrame(columns=['class', 'f1-score', 'AUC', 'ACC', 'TP', 'FP', 'TN', 'FN'])
    print(list(name_dict.values()))
    print(len(mcof_mat))
    aucs = []
    for c, cm in enumerate(mcof_mat):

        name_cls = f'{list(name_dict.values())[c]}'
        each_auc = roc_auc_score(y_test[:, c], test_prediction[:, c], multi_class="ovr")
        each_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
        print(f'confusion matrix: {name_cls}')
        print(cm)
        print(f'AUC: {each_auc}')
        cat_cm = pd.DataFrame({'class': name_cls,
                                'f1-score': f'{f1_each[c]:0.2f}',
                                'AUC': f'{each_auc:0.2f}',
                                'ACC': f'{each_acc:0.2f}',
                                'TN': cm[0][0],
                                'FP': cm[0][1],
                                'FN': cm[1][0],
                                'TP': cm[1][1]}, index=[name_cls])
        mcmList = pd.concat([mcmList, cat_cm], axis = 0)
        aucs.append(each_auc)
    mcmList.to_excel(os.path.join(path_savemodel, f'{clf_name}_Multi-ConfusionMatrix_{date}.xlsx'))
    cof_mat = confusion_matrix(y_test.argmax(axis=1), test_prediction.argmax(axis=1))
    cof_mat = pd.DataFrame(cof_mat, columns=list(name_dict.values()), index = list(name_dict.values()))
    cof_mat = pd.concat([cof_mat, pd.DataFrame({'f1_score_macro':f1_score(y_test.argmax(axis=1), test_prediction.argmax(axis=1), average='macro'),
                                                    'AUC_macro':f'{sum(aucs)/len(aucs):0.3f}',
                                                    'accurcy': f'{acc_score:0.3f}',
                                                    }, index = ['Total'])])
    cof_mat.to_excel(os.path.join(path_savemodel, f'{clf_name}_ConfusionMatrix_{date}.xlsx'))


    # probclassList = [f'prob_{name_dict[name]}' for name in list(name_dict.keys())]
    # probList = pd.DataFrame(proba, columns=probclassList)

    # resultList = pd.DataFrame(columns=['file_name', 'GT', 'Prediction'])
    # fail_idx = []
    # i = 0
    # for gt, yp in zip(np.array(y_test), test_prediction.argmax(axis=1)):
    #     gt_name = name_dict[gt]
    #     pr_name = name_dict[yp]
    #     sub_df = pd.DataFrame({'file_name': f'{np.array(f_test)[i]}', 
    #                        'GT': gt_name, 
    #                        'Prediction':pr_name,
    #                        }, index=[i])
    #     resultList = pd.concat([resultList, sub_df])

    #     if gt != yp:
    #         fail_idx.append(i)


    #     i+=1
    # resultList = pd.concat([resultList, probList], axis=1)
    # failList = resultList.loc[fail_idx]
    # failList.to_excel(os.path.join(path_savemodel, f'{clf_name}_FailList_{date}.xlsx'))
    # resultList.to_excel(os.path.join(path_savemodel, f'{clf_name}_ResultList_{date}.xlsx'))
    # return proba, probList, resultList


def drawing_ROCcurve(y_test, test_result, savepath, clf_name, name_dict):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.metrics import roc_curve,auc


    # test_y = tf.keras.utils.to_categorical(y_test)
    test_y = y_test[:,list(name_dict.values())]
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
    for nc, color in zip(list(name_dict.values()), colors[:n_classes]):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="class {0} (auc = {1:0.2f})".format(list(name_dict.keys())[nc], roc_auc[i]),
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
    plt.savefig(f'{savepath}/ROCcurve_{clf_name}_{n_classes}class.png')

    return fpr["micro"], tpr["micro"], roc_auc["micro"], fpr["macro"], tpr["macro"], roc_auc["macro"]