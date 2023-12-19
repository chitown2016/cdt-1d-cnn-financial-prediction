
import strategy_development.cnn.cnn2.prepare_data as prep
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Activation
import numpy as np
import pickle


def create_model_cnn(**kwargs):

    # now running 1 do

    conv2d_strides = 1
    kernel_regularizer = 1e-5
    adam_initial_learning_rate = 1e-3
    dropout_rate = 0.5  # 0.3 in the benchmark version
    model = Sequential()

    conv2d_layer1 = Conv2D(32, (1,4), strides = conv2d_strides,
                           kernel_regularizer=regularizers.l2(kernel_regularizer),
                           padding='same', use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(5,24,1))


    model.add(conv2d_layer1)
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(1,4), strides=(1,4), padding='valid'))
    #model.add(Dropout(dropout_rate)) # this one was commented out in the benchmark version

    conv2d_layer2 = Conv2D(64, (1, 3), strides=conv2d_strides,
                           kernel_regularizer=regularizers.l2(kernel_regularizer),
                           padding='same', use_bias=True,
                           kernel_initializer='glorot_uniform')
    
    model.add(conv2d_layer2)
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1,3), padding='valid'))
    #model.add(Dropout(dropout_rate)) # this one was commented out in the benchmark version


    conv2d_layer3 = Conv2D(128, (1, 2), strides=conv2d_strides,
                           kernel_regularizer=regularizers.l2(kernel_regularizer),
                           padding='same', activation='relu', use_bias=True,
                           kernel_initializer='glorot_uniform')

    model.add(conv2d_layer3)
    model.add(MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))
    #model.add(Dropout(dropout_rate)) # this one was commented out in the benchmark version

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    #model.add(Dropout(dropout_rate)) # this one was commented out in the benchmark version
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=adam_initial_learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=[tf.keras.metrics.CategoricalAccuracy()])


    return model



def main_run():
    final_data = prep.prepare_2H_data()

    feature_data = prep.get_features(df=final_data)

    model = create_model_cnn()

    pred_classes_list = []

    data_path = '/workspaces/PycharmProjects/strategy_development/cnn/cnn2/model_output'
    #data_path = 'C:\Research\PycharmProjects/strategy_development/cnn/cnn2/model_weights'
    file_name = 'cnn_data_final.pkl'
    print(os.getcwd())

    if os.path.isfile(data_path + '/cnn2_weights.h5'):
        model.load_weights(data_path + '/cnn2_weights.h5')
    else:
        model.save_weights(data_path + '/cnn2_weights.h5')

    
    rolling_indices_output = prep.prepare_rolling_simulation_indices(entire_data=final_data)

    train_start_index_list = rolling_indices_output['train_start_index_list']
    train_end_index_list = rolling_indices_output['train_end_index_list']

    test_start_index_list = rolling_indices_output['test_start_index_list']
    test_end_index_list = rolling_indices_output['test_end_index_list']

    for i in range(len(train_start_index_list)):

        
        train_start_index = train_start_index_list[i]
        train_end_index = train_end_index_list[i]

        test_start_index = test_start_index_list[i]
        test_end_index = test_end_index_list[i]

        x_train_i = feature_data.iloc[train_start_index:train_end_index, :]
        y_train_i = np.array(final_data['label'].iloc[train_start_index:train_end_index])

        if test_end_index == -1:
            x_test_i = feature_data.iloc[test_start_index:, :]
            y_test_i = np.array(final_data['label'].iloc[test_start_index:])
        else:
            x_test_i = feature_data.iloc[test_start_index:test_end_index, :]
            y_test_i = np.array(final_data['label'].iloc[test_start_index:test_end_index])

        scaler_i = StandardScaler()
        x_train_i = scaler_i.fit_transform(x_train_i)
        x_test_i = scaler_i.transform(x_test_i)

        x_train_i = prep.reshape_data(x_train_i)
        x_test_i = prep.reshape_data(x_test_i)

        trained_weights_file = data_path + '/cnn2_' + str(train_start_index) + '_weights.h5'
        
        if not os.path.isfile(trained_weights_file):

            one_hot_enc = OneHotEncoder(sparse_output=False, categories=[[0,1,2]])  # , categories='auto'
            y_train_i = one_hot_enc.fit_transform(y_train_i.reshape(-1, 1))
            y_test_i = one_hot_enc.fit_transform(y_test_i.reshape(-1, 1))

            model.load_weights(data_path + '/cnn2_weights.h5')
            
            history = model.fit(x_train_i, y_train_i, epochs=200, verbose=1,  # params['epochs']
                    batch_size=64, shuffle=True, validation_data=(x_test_i, y_test_i)) # benchmark bath_size=256
            model.save_weights(trained_weights_file)

            with open(data_path + '/cnn2_' + str(train_start_index) + '_history.pickle', 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

            pred = model.predict(x_test_i)
            pred_classes = np.argmax(pred, axis=1)
            pred_classes_list.append(pred_classes)

            print('Percent Completed: ' + str(100 * i / len(train_start_index_list)))
            print(10*'*')
        else:
            model.load_weights(trained_weights_file)
            pred = model.predict(x_test_i)
            pred_classes = np.argmax(pred, axis=1)
            pred_classes_list.append(pred_classes)

    final_data['predicted'] = np.nan
    final_data['predicted'].iloc[test_start_index_list[0]:] = np.concatenate(pred_classes_list)
    final_data.to_pickle(file_name)

if __name__ == "__main__":
    main_run()




















if __name__ == "__main__":
    main_run()