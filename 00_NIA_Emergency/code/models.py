import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Embedding, LSTM, Bidirectional

def model_mlp(n_input_dim, n_unit, n_class):
    
    unitList = [n_input_dim] + n_unit + [n_class]
    
    inputs = Input((n_input_dim,))
    
    for nu, nunit in enumerate(unitList[1:-1]):
        x = Dense(unitList[nu+1], #################
                        input_dim=nunit,
                        kernel_initializer=tf.keras.initializers.HeNormal(),
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0.0001),
                        # bias_regularizer=regularizers.L1L2(l1=0, l2=0.001),
                        activation='relu')(inputs)
        
    outputs = Dense(n_class, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def model_BiLSTM(n_input_dim, emb_dim, hidden_units, n_class):
    model = Sequential()
    model.add(Embedding(input_dim = n_input_dim,
                        output_dim = emb_dim))
    model.add(Bidirectional(LSTM(hidden_units,
                                dropout=0.3))) # Bidirectional LSTM을 사용
    model.add(Dense(n_class, activation='softmax'))
    return model