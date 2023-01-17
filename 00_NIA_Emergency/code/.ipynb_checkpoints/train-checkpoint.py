import os, sys
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import json
import numpy as np
import pandas as pd
import datetime

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

import utils
import models as ai_model



class _get_tfidf_ml:
    def __init__(self, configs, df_all):
        self.acceptList = configs["acceptList"]
        self.stop_words = configs["Stop_word"]
        self.df_all = df_all
        self.configs = configs

    def _get_corpus(self, x_train, y_train, f_train):
        X_train_token = utils.pos_select(x_train, y_train, f_train, self.stop_words, self.acceptList)
        corp_Xtrain = utils._making_token(X_train_token)
        Y_train = tf.keras.utils.to_categorical(y_train)

        return corp_Xtrain, Y_train

    def _get_tfidfVec(self, mlp_par):

        X_train, x_test, Y_train, y_test, F_train, f_test= train_test_split(self.df_all['context'], self.df_all['class'], 
                                                                            self.df_all['file_name'], 
                                                                            test_size = 0.1, 
                                                                            random_state = self.configs["random_seed"], 
                                                                            stratify= self.df_all['class'])
        x_train, x_val, y_train, y_val, f_train, f_val = train_test_split(X_train, Y_train, F_train, 
                                                                            test_size = 1/9, 
                                                                            random_state = self.configs["random_seed"], 
                                                                            stratify=Y_train)

        
        corp_Xtrain, Y_train = self._get_corpus(x_train, y_train, f_train)
        corp_Xval, Y_val = self._get_corpus(x_val, y_val, f_val)
        corp_Xtest, Y_test = self._get_corpus(x_test, y_test, f_test)


        vectorizer = TfidfVectorizer(max_features=mlp_par["max_features"],ngram_range=(1,mlp_par["max_ngram"]), min_df=1)
        
        X_train = vectorizer.fit_transform(corp_Xtrain).toarray()
        X_val = vectorizer.transform(corp_Xval).toarray()
        X_test = vectorizer.transform(corp_Xtest).toarray()

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, f_test




def main():
    ######################
    # Setting parameters
    ######################
    s = datetime.datetime.now().strftime('%Y-%m-%d %H:%M: %S')
    print(s)
    ## Setting class
    class_dict = {'01.예진':0,
                '02.초진':1,
                '03.투약및검사':2,
                '04.검사결과설명및퇴실':3}
    n_class = len(class_dict)

    ## Setting default path
    path_dataDF = os.path.join(os.path.abspath('..'), 'data')
    path_saveDF = os.path.join(os.path.abspath('..'), 'model')
    os.makedirs(os.path.join(path_saveDF, 'result'), exist_ok=True)

    ## Loading configuration file
    with open('config.json', "r", encoding="UTF-8") as f:
        configs = json.load(f)

    ## Setting configuration
    random_seed = configs["random_seed"]
    epochs = configs["epochs"]
    batch = configs["batch"]
    LR = configs["LR"]
    mlp_par = configs["mlp_parameter"]

    
    
    ######################
    # Loading data
    ######################
    print('Loading data...')
    df_all = utils.preprocessing_data2morph(path_dataDF, class_dict)
    
    ## tf-idf vectorazation
    cls_tfidf = _get_tfidf_ml(configs, df_all)
    X_train, Y_train, X_val, Y_val, X_test, Y_test, F_test = cls_tfidf._get_tfidfVec(mlp_par)

    ######################
    # Training model
    ######################

    ## Loading a model structure
    
    model = ai_model.model_mlp(n_input_dim=mlp_par["max_features"], n_unit=mlp_par["n_units"], n_class=n_class)
    plot_model(model, show_shapes=True)

    ## the model compile
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])
    ## Setting callbacks
    checkpointer = ModelCheckpoint(filepath=os.path.join(path_saveDF,'checkpoint','cp_{epoch:02d}.ckpt'), save_weights_only=True,
                                    verbose=1, monitor='val_loss', period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=10, min_lr=0, min_delta=1e-10, verbose=1)
    
    EarlyStopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    callbacks_list = [checkpointer, reduce_lr, EarlyStopper]
    
    ## training the model
    with tf.device("/device:GPU:0"):
        training_model = model.fit(x= X_train, y= Y_train, 
                                batch_size=batch, 
                                validation_data = (X_val,Y_val), 
                                epochs= epochs, 
                                verbose=1, 
                                shuffle=True, 
                                callbacks=callbacks_list)
    
    ## Saving a last weight
    model.save_weights(os.path.join(path_saveDF,f'last_model.h5'))

    ######################
    # Result prediction
    ######################
    print('\n'+'#'*(len('Accuracy Training data: ')+5))
    print('Results')
    predicted_train_tfidf = model.predict(X_train,verbose=0)
    predicted_test_tfidf = model.predict(X_test,verbose=0)

    accuracy_train_tfidf = accuracy_score(Y_train.argmax(axis=1), predicted_train_tfidf.argmax(axis=1))
    accuracy_test_tfidf = accuracy_score(Y_test.argmax(axis=1), predicted_test_tfidf.argmax(axis=1))
    print('Accuracy Training data: {:.1%}'.format(accuracy_train_tfidf))
    print('Accuracy Test data: {:.1%}'.format(accuracy_test_tfidf))
    print('#'*(len('Accuracy Training data: ')+5))
    cof_mat = confusion_matrix(Y_test.argmax(axis=1), predicted_test_tfidf.argmax(axis=1))
    cof_mat = pd.DataFrame(cof_mat, columns=list(class_dict.keys()), index = list(class_dict.keys()))
    cof_mat.to_excel(os.path.join(path_saveDF, 'result', f'ConfusionMatrix.xlsx'))

    ## Saving each result        
    resultList = pd.DataFrame(columns=['File name', 'GT', 'Prediction', 'Result'])
    i=0
    for gt, yp in zip(Y_test.argmax(axis=1), predicted_test_tfidf.argmax(axis=1)):
        gt_name = list(class_dict.keys())[gt]
        pr_name = list(class_dict.keys())[yp]
        if gt != yp:
            match = 'F'
        else:
            match = 'T'
        sub_df = pd.DataFrame({'File name': f'{np.array(F_test)[i]}', 
                            'GT': gt_name, 
                            'Prediction':pr_name,
                            'Result': match,
                            }, index=[i])
        resultList = pd.concat([resultList, sub_df])
        i+=1
    resultList.to_excel(f'ResultList.xlsx')
    ## Saving meta graph
    import tensorflow.compat.v1 as tf1
    tf1.disable_v2_behavior()
    tf1.compat.v1.train.export_meta_graph(filename=os.path.join(path_saveDF,'checkpoint','metagraph.meta'),
                                        collection_list=["input_tensor", "output_tensor"])
    print('All process done!')
    s = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(s)

if __name__ == '__main__':
    main()
    
