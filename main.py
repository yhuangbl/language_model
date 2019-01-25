from __future__ import print_function
import argparse
import json
import os
import time
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, concatenate, add, merge
from keras.layers import GRU, Bidirectional, LSTM, PReLU, Dropout, Average
from keras.layers import Activation, Conv1D, Flatten, Lambda, TimeDistributed
from keras.layers import MaxPooling1D, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import numpy as np
from data_helper import load_data, build_input_data
from scorer import scoring
from utils import TestCallback, make_submission


def build_model(model, embedding_dim, hidden_size, drop, num_filter, sequence_length, vocabulary_size):
    if model == "cnn":
        return build_cnn_model(embedding_dim, drop, num_filter, sequence_length, vocabulary_size)
    elif model == "lstm1":
        return build_lstm1_model(embedding_dim, hidden_size, drop, sequence_length, vocabulary_size)
    elif model == "lstm2":
        return build_lstm2_model(embedding_dim, hidden_size, drop, sequence_length, vocabulary_size)
    elif model == "gru1":
        return build_gru1_model(embedding_dim, hidden_size, drop, sequence_length, vocabulary_size)
    elif model == "cnn_lstm":
        return build_cnn_lstm_model(embedding_dim, hidden_size, drop, num_filter, sequence_length, vocabulary_size)


def build_save_ensemble_model(saved_model, models, sequence_length):
    build_ensemble_avg_model(saved_model, models, sequence_length)


def build_ensemble_avg_model(saved_model, models, sequence_length):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    outputs = [model(inputs) for model in models]
    ensemble_output = Average()(outputs)
    
    model = Model(inputs, ensemble_output, name='ensemble_avg')
    print(model.summary())
    model.save(saved_model)

                                      
def build_cnn_model(embedding_dim, drop,
                    num_filter, sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    embedding = emb_layer(inputs)
    drop_embed = Dropout(drop)(embedding)
    
    conv1 = Conv1D(num_filter, 3, activation='relu', padding='same')(drop_embed)
    conv1 = MaxPooling1D(3, padding='same', data_format='channels_first')(conv1)
    conv2 = Conv1D(num_filter, 4, activation='relu', padding='same')(drop_embed)
    conv2 = MaxPooling1D(3, padding='same', data_format='channels_first')(conv2)
    conv3 = Conv1D(num_filter, 5, activation='relu', padding='same')(drop_embed)
    conv3 = MaxPooling1D(3, padding='same', data_format='channels_first')(conv3)

    conc = concatenate([conv1, conv2, conv3])
    outputs = Dense(units=vocabulary_size, activation='softmax')(conc)
 
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def build_lstm1_model(embedding_dim, hidden_size, drop,
                      sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    embedding = emb_layer(inputs)
    drop_embed = Dropout(drop)(embedding)
    
    lstm_out_1 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(drop_embed)
    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(lstm_out_1)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def build_lstm2_model(embedding_dim, hidden_size, drop,
                      sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    embedding = emb_layer(inputs)
    drop_embed = Dropout(drop)(embedding)
    
    lstm_out_1 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(drop_embed)
    lstm_out_2 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(lstm_out_1)
    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(lstm_out_2)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def build_gru1_model(embedding_dim, hidden_size, drop,
                    sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    embedding = emb_layer(inputs)
    drop_embed = Dropout(drop)(embedding)
    
    gru_out_1 = GRU(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(drop_embed)
    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(gru_out_1)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def build_cnn_lstm_model(embedding_dim, hidden_size, drop,
                         num_filter, sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    embedding = emb_layer(inputs)
    drop_embed = Dropout(drop)(embedding)
    
    lstm_out_1 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(drop_embed)
    lstm_out_2 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(lstm_out_1)
    
    block1 = Conv1D(num_filter, 3, padding='same')(drop_embed)
    block1 = PReLU()(block1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling1D(3, padding='same', data_format='channels_first')(block1)
    block1 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(block1)
    block1 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(block1)
    
    block2 = Conv1D(num_filter, 4, padding='same')(drop_embed)
    block2 = PReLU()(block2)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling1D(4, padding='same', data_format='channels_first')(block2)
    block2 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(block2)
    block2 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(block2)
    
    block3 = Conv1D(num_filter, 5, padding='same')(drop_embed)
    block3 = PReLU()(block3)
    block3 = BatchNormalization()(block3)
    block3 = MaxPooling1D(5, padding='same', data_format='channels_first')(block3)
    block3 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(block3)
    block3 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(block3)
    
    conc = concatenate([lstm_out_2, block1, block2, block3])
    
    conc_dense = Dense(units=100, activation='relu')(conc)
    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(conc_dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def predict_final_word(model, vocabulary, filename):
    id_list = []
    prev_tokens_list = []
    prev_tokens_lens = []
    with open(filename, "r") as fin:
        fin.readline()
        for line in fin:
            id_, prev_sent, grt_last_token = line.strip().split(",")
            id_list.append(id_)
            prev_tokens = prev_sent.split()
            prev_tokens_list.append(prev_tokens)
            prev_tokens_lens.append(len(prev_tokens))
    X = np.array([build_input_data(t, vocabulary)[0][0].tolist()
                  for t in prev_tokens_list])
    y_prob = model.predict(X, batch_size=32)
    last_token_probs = np.array([y_prob[b, prev_tokens_lens[b] - 1, :]
                                 for b in range(y_prob.shape[0])])

    return dict(zip(id_list, last_token_probs))


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    np.random.seed(opt.seed) # set a seed for reproduciaiblity
    if opt.mode == "train":
        st = time.time()
        print('Loading data')
        x_train, y_train, x_valid, y_valid, vocabulary_size = load_data(
            "data", opt.debug)

        num_training_data = x_train.shape[0]
        sequence_length = x_train.shape[1]
        print(num_training_data)

        print('Vocab Size', vocabulary_size)

        model = build_model(opt.model, opt.embedding_dim, opt.hidden_size, opt.drop,
                            opt.filter, sequence_length, vocabulary_size)
        adam = Adam()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
        print("Traning Model...")
        checkpoint = ModelCheckpoint(opt.saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        history = model.fit(x_train, y_train, batch_size=opt.batch_size,
                            epochs=100, verbose=1, validation_data=(x_valid, y_valid),
                            callbacks=[TestCallback((x_valid,y_valid), model=model), checkpoint, early])
        model.save(opt.saved_model)
        print("Training cost time: ", time.time() - st)
    elif opt.mode == "ensemble":
        x_train, y_train, x_valid, y_valid, vocabulary_size = load_data(
            "data", opt.debug)

        num_training_data = x_train.shape[0]
        sequence_length = x_train.shape[1]
        print(num_training_data)

        print('Vocab Size', vocabulary_size)
        
        ENSEMBLE_DIR = "models/ensemble/"
        model_files = []
        for (dirpath, dirnames, filenames) in os.walk(ENSEMBLE_DIR):
            model_files.extend(filenames)
            break
        models = []
        model_count = 0
        for filename in model_files:
            model = load_model(ENSEMBLE_DIR + filename)
            model.name="model" + str(model_count)
            model_count += 1
            models.append(model)
        
        build_save_ensemble_model(opt.saved_model, models, sequence_length)
    else:
        model = load_model(opt.saved_model)
        vocabulary = json.load(open(os.path.join("data", "vocab.json")))
        predict_dict = predict_final_word(model, vocabulary, opt.input)
        sub_file = make_submission(predict_dict, opt.student_id, opt.input)
        if opt.score:
            scoring(sub_file, os.path.join("data"), type="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="train", choices=["train", "test", "ensemble"],
                        help="Train, ensemble or test mode")
    parser.add_argument("-saved_model", type=str, default="model.h5", help="saved model path")
    parser.add_argument("-input", type=str, default=os.path.join("data", "valid.csv"),
                        help="Input path for generating submission")
    parser.add_argument("-debug", action="store_true",
                        help="Use validation data as training data if it is true")
    parser.add_argument("-score", action="store_true", help="Report score if it is")
    parser.add_argument("-student_id", default=None, required=True,
                        help="Student id number is compulsory!")

    parser.add_argument("-batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("-embedding_dim", type=int, default=100, help="word embedding dimension")
    parser.add_argument("-hidden_size", type=int, default=500, help="rnn hidden size")
    parser.add_argument("-drop", type=float, default=0.5, help="dropout")
    parser.add_argument("-filter", type=int, default=64, help="number of filters")
    parser.add_argument("-gpu", type=str, default="", help="gpu")
    parser.add_argument("-seed", type=int, default=1122, help="numpy seed")
    parser.add_argument("-model", type=str, default="",
                        choices=["lstm1", "lstm2", "cnn", "gru1", "cnn_lstm"], help="model type")
    opt = parser.parse_args()
    main(opt)
