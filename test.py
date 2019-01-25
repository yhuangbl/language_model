''' This module contains utility functions and global variables for Keras machine learning model. '''
from collections import defaultdict
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold

from emotionnet import emotion_tokenizer
from utils import pandas_utils, utils_func

# global variables
NUM_WORDS = 15000  # based on the results we got in playground.ipynb
emotion_cat = {"anger": 0, "fear": 1, "joy": 2, "sadness": 3}
MAX_LEN = 50  # based on the number we get in playground.ipynb (maximum number of words 32)
NUM_CLASSES = 4
FYT_DIR = os.environ["REBECCA_FYT"]
EMOJI_NUM = 1661

from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


losses = {"category_output": "categorical_crossentropy",
          "intensity_output": "mean_squared_error"}
compilers = {"cateogry_output": "adam",
             "intensity_output": "rmsprop"}


def get_train_test():
    """ Gets the training and testing data.

    Returns:
        dict: the returned dictionary has three keys: train, test, word_index. dict["train"] gives the training data and
        the correct emotion label and emotion intensity for the training data. dict["test"] gives the testing data and
        the correct emotion label and emotion intensity for the testing data. dict["word_index"] gives the unique words
        found in training data and the id(i.e. the index) for each word.

    """
    train = pandas_utils.read_dataset(FYT_DIR + "/data/train/all.txt")
    test = pandas_utils.read_dataset(FYT_DIR + "/data/test/all.txt")
    train_texts = train["tweet"].tolist()
    test_texts = test["tweet"].tolist()

    # Tokenization
    tokenizer = emotion_tokenizer.EmotionTokenizer()
    tokenizer.fit_on_texts(train_texts)
    sequences_train = tokenizer.texts_to_sequences(train_texts)
    sequences_test = tokenizer.texts_to_sequences(test_texts)
    word_index = tokenizer.word_index
    print("Found {} unique tokens.".format(len(word_index)))

    # Transform emotion category from word to int
    train_labels = train.emotion.apply(lambda x: emotion_cat[x])
    test_labels = test.emotion.apply(lambda x: emotion_cat[x])

    # Generate X, y numpy arraies
    X_train = pad_sequences(sequences_train, maxlen=MAX_LEN)
    y_cat_train = np.asarray(train_labels)
    y_int_train = np.asarray(train["score"])
    X_test = pad_sequences(sequences_test, maxlen=MAX_LEN)
    y_cat_test = np.asarray(test_labels)
    y_int_test = np.asarray(test["score"])

    return {"train": (X_train, y_cat_train, y_int_train),
            "test": (X_test, y_cat_test, y_int_test),
            "word_index": word_index}


# best epoch is the epoch that achieves the lowest validation loss
def get_best_epoch(history_obj):
    val_loss = history_obj["val_loss"]
    min_val_loss = min(val_loss)
    return val_loss.index(min_val_loss) + 1  # epoch starts with 1


def run_kfold(n_splits, model_func, X, y, weight_f):
    """ Runs Stratified k-fold cross validation on a machine learning model.

    Args:
        n_splits (int): number of folds.
        model_func (func): function to get the machine learning model.s
        X (numpy array): training data.
        y (numpy array): correct emotion labels and emotion intensity values.

    Returns:
        list: The first item in the list is the cross validation score for each fold in numpy array format. The second
        item in the list is the average cross validation score in string format.

    """
    y_cat = y[0]
    y_int = y[1]

    kf = StratifiedKFold(n_splits=n_splits)
    fold_id = -1
    cv_scores = []
    # history_objs = []
    y_cat_one_hot = to_categorical(y_cat, num_classes=NUM_CLASSES)
    best_epochs = []

    two_inputs = False
    if type(X) is list:
        X = X[0]
        two_inputs = True

    print("Overall intensity distribution")
    utils_func.draw_dist(y_int)

    for train_idx, val_idx in kf.split(X, y_cat):
        fold_id += 1

        model = model_func()
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        model.compile(optimizer="adam", loss=losses)

        X_train, X_val = X[train_idx], X[val_idx]
        if two_inputs:
            X_train = [X_train, X_train]
            X_val = [X_val, X_val]
        y_cat_train, y_cat_val = y_cat_one_hot[train_idx], y_cat_one_hot[val_idx]
        y_int_train, y_int_val = y_int[train_idx], y_int[val_idx]

        print("Fold {}".format(fold_id))
        print("Fold {} intensity distribution (train & val)".format(fold_id))
        utils_func.draw_dist(y_int_train)
        utils_func.draw_dist(y_int_val)
        # checkpoint
        filepath = weight_f
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        hist = model.fit(X_train, {"category_output": y_cat_train, "intensity_output": y_int_train},
                         validation_data=(X_val, {"category_output": y_cat_val, "intensity_output": y_int_val}),
                         epochs=50, verbose=0, callbacks=[checkpoint, early])
        best_epochs.append(get_best_epoch(hist.history))
        model.load_weights(filepath)
        cv_scores.append(model.evaluate(X_val, {"category_output": y_cat_val, "intensity_output": y_int_val}))

        cv_array = np.asarray(cv_scores)
        cv_mean = np.mean(cv_array, axis=0)
        metrics_names = model.metrics_names
        average_info = ["Average CV result:"]
        for i in range(len(metrics_names)):
            average_info.append("val_{}: {}".format(metrics_names[i], cv_mean[i]))
        print(best_epochs)
        print(cv_array)
    return cv_scores, average_info, sum(best_epochs) / len(best_epochs)


def train_model(model_func, X_train, y_train, avg_epochs):
    epochs = round(avg_epochs)
    y_cat_train = to_categorical(y_train[0], num_classes=NUM_CLASSES)
    y_int_train = y_train[1]

    model = model_func()
    model.compile(optimizer="adam", loss=losses)
    hist = model.fit(X_train, {"category_output": y_cat_train, "intensity_output": y_int_train},
                     epochs=epochs, verbose=1)
    return model, hist


def avg(a_list):
    return sum(a_list) / len(a_list)


def eval_result(actual, pred):
    """ Evaluates the results predicted by a machine learning model. This function will evaluate the result for emotion
    category classification and emotion intensity measurement.

    Reference for multiclass roc auc score:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings

    Args:
        actual (numpy array): the ground truth.
        pred (numpy array): the predictions made by the machine learning model.
    """
    true_cat_sparse = actual["cat"]
    pred_cat_sparse = pred["cat"]
    print("===== Evaluate category result =====")
    roc_auc_scores = []
    for i in range(NUM_CLASSES):
        roc_auc = eval_cat_result(true_cat_sparse[:, i], pred_cat_sparse[:, i], i)
        roc_auc_scores.append(roc_auc)
    print("\nAverage ROC AUC: {}".format(sum(roc_auc_scores) / len(roc_auc_scores)))

    true_cat = np.argmax(true_cat_sparse, axis=1)
    pred_cat = np.argmax(pred_cat_sparse, axis=1)

    true_cat_len = len(true_cat)
    assert true_cat_len == len(pred_cat)
    match_idx = defaultdict(list)
    for i in range(true_cat_len):
        if true_cat[i] == pred_cat[i]:
            match_idx[true_cat[i]].append(i)

    true_int = actual["int"].tolist()
    pred_int = pred["int"].tolist()
    print("===== Evaluate intensity result =====")
    r_coefs = []
    r_large_coefs = []
    mses = []
    mse_larges = []
    for cat, idx in match_idx.items():
        # only compare the intensity values that the emotion categories are correctly classified
        true_cat_int = [true_int[i] for i in idx]
        pred_cat_int = [pred_int[i] for i in idx]
        r, r_large, mse, mse_large = eval_int_result(true_cat_int, pred_cat_int, cat)
        r_coefs.append(r)
        r_large_coefs.append(r_large)
        mses.append(mse)
        mse_larges.append(mse_large)
    print("\nAverage Pearson correlation: {}".format(avg(r_coefs)))
    print("Average Pearson correlation for gold scores in range 0.5-1: {}".format(avg(r_large_coefs)))
    print("Average mean squared error: {}".format(avg(mses)))
    print("Average mean squared error for gold scores in range 0.5-1: {}".format(avg(mse_larges)))


def eval_cat_result(actual, pred, cat):
    """ Evaluates the emotion category classification results predicted by a machine learning model.

    Args:
        actual (numpy array): the ground truth.
        pred (numpy array): the predictions made by the machine learning model.
        cat (int): emotion category id.
    """
    cat_text = utils_func.get_key_from_value(emotion_cat, cat)
    assert cat_text is not None
    print("{}:".format(cat_text))
    roc_auc = roc_auc_score(actual, pred)
    print("ROC AUC score: {}".format(roc_auc))
    return roc_auc


def eval_int_result(actual, pred, cat):
    """ Evaluates the emotion intensities measurement results predicted by a machine learning model.

    Args:
        actual (numpy array): the ground truth.
        pred (numpy array): the predictions made by the machine learning model.
        cat (int): emotion category id.
    """
    cat_text = utils_func.get_key_from_value(emotion_cat, cat)
    assert cat_text is not None
    print("{}:".format(cat_text))
    r, p_value = pearsonr(actual, pred)
    mse = mean_squared_error(actual, pred)
    print("Pearson correlation: {}\nmean squared error: {}".format(r, mse))

    large_idx = {i for i in range(len(actual)) if actual[i] >= 0.5}
    large_actual = [actual[i] for i in large_idx]
    large_pred = [pred[i] for i in large_idx]
    r_large, p_value_large = pearsonr(large_actual, large_pred)
    mse_large = mean_squared_error(large_actual, large_pred)
    print("Pearson correlation for gold scores in range 0.5-1: {}".format(r_large))
    print("mean squared error for gold score in range 0.5-1: {}\n".format(mse_large))
    return r, r_large, mse, mse_large
