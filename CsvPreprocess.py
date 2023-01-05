import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import category_encoders as ce

def scale_features(f):
    f_mean = np.mean(f, axis=0)
    f_min_max = np.max(f, axis=0) - np.min(f, axis=0)
    return (f - f_mean) / f_min_max


def get_features(inp, oup):
    cbe = ce.cat_boost.CatBoostEncoder()
    cbe.fit(inp, oup)
    inp = cbe.transform(inp)
    inp = inp.to_numpy()
    inp = np.round(inp.astype(float), decimals=4)
    # CatBoostEncoder encodes NaNs for string inputs. 
    # For numeric inputs it leaves them untouched (weight, height).
    # Replace any remaining NaNs with 0.0
    inp = np.nan_to_num(inp, nan=0.0)
    inp = scale_features(inp)
    return inp

def multiClassEncode(y):
    # Number of unique elements in y is number of classes
    num_classes = len(set(y))
    output = np.zeros((len(y), num_classes))
    output[np.arange(len(y)), y] = 1
    return output

def nan_count(arr):
    nan_arr = np.isnan(arr)
    return (nan_arr == True).sum()

def assert_nan_count(arrays):
    for idx, arr in enumerate(arrays):
        assert nan_count(arr) == 0, "nan count for arr } = {}".format(nan_count(arr))

def split_train_test(inp, oup):
    f_train, f_test, l_train, l_test = train_test_split(inp, oup, test_size=0.1) # use 10% for testing
    f_train = np.round(f_train.astype(float), decimals=4)
    f_test = np.round(f_test.astype(float), decimals=4)
    assert_nan_count([f_train, f_test, l_train, l_test]) 
    return f_train, f_test, l_train, l_test

def get_data():
    df = pd.read_csv('training_data.csv')

    features = df.iloc[:, :-3]
    murmurs = df.loc[:, 'Murmur']
    outcomes = df.loc[:, 'Outcome']

    # Encode murmurs
    murmurs = murmurs.to_numpy()
    le = preprocessing.LabelEncoder()
    le.fit(murmurs)
    murmurs = le.transform(murmurs)

    # Encode outputs
    outcomes = outcomes.to_numpy()
    le = preprocessing.LabelEncoder()
    le.fit(outcomes)
    outcomes = le.transform(outcomes)

    # Feature set is same. However CatBoostEncoder takes labels into consideration 
    # (see cbe.fit above) for encoding. As a result we end up having 2 sets of features
    # each named after the label
    features_murmurs = get_features(features, murmurs)
    features_outcomes = get_features(features, outcomes)

    murmurs = multiClassEncode(murmurs)
    outcomes = multiClassEncode(outcomes)
    np.isnan(features_murmurs)

    # Get training and testing data for Murmur Classifier (mc)
    mci_train, mci_test, murmurs_train, murmurs_test = split_train_test(features_murmurs, murmurs)
    # Get training and testing data for Outcomes Classifier (oc)
    oci_train, oci_test, outcomes_train, outcomes_test = split_train_test(features_outcomes, outcomes)
    return mci_train, mci_test, oci_train, oci_test, murmurs_train, murmurs_test, outcomes_train, outcomes_test

def get_data_2():
    df = pd.read_csv('training_data.csv')

    df.drop('Patient ID', inplace=True, axis=1)
    features = df.iloc[:, :-3]
    features.drop('Murmur', inplace=True, axis=1)
    murmurs = df.loc[:, 'Murmur']
    outcomes = df.loc[:, 'Outcome']

    murmurs = murmurs.to_numpy()
    outcomes = outcomes.to_numpy()

    le = preprocessing.LabelEncoder()
    le.fit(murmurs)
    murmurs = le.transform(murmurs)
    murmurs_1_nan_index = np.where(le.classes_ == 'Unknown')
    if murmurs_1_nan_index[0].size == 0:
        murmurs_1_nan_index = None
    else:
        murmurs_1_nan_index = murmurs_1_nan_index[0][0]

    le = preprocessing.LabelEncoder()
    le.fit(outcomes)
    outcomes = le.transform(outcomes)
    outcomes_nan_index = np.where(le.classes_ == 'Unknown')
    if outcomes_nan_index[0].size == 0:
        outcomes_nan_index = None
    else:
        murmurs2_nan_index = outcomes_nan_index[0][0]
        
    cbe = ce.cat_boost.CatBoostEncoder()
    cbe.fit(features, murmurs)
    features_1 = cbe.transform(features)
    features_1.iloc[875, :]

    cbe = ce.cat_boost.CatBoostEncoder()
    cbe.fit(features, outcomes)
    features_2 = cbe.transform(features)
    features_2.iloc[875, :]

    def encodeLabels(y, numClasses):
        output = np.zeros((len(y), numClasses))
        output[np.arange(len(y)), y] = 1
        return output

    murmurs = encodeLabels(murmurs, 3)
    outcomes = encodeLabels(outcomes, 2)

    features_train_1, features_test_1, murmurs_train, murmurs_test = train_test_split(features_1, murmurs, test_size=0.1)
    features_train_2, features_test_2, outcomes_train, outcomes_test = train_test_split(features_2, outcomes, test_size=0.1)
    features_train_1 = features_train_1.to_numpy()
    features_train_2 = features_train_2.to_numpy()

    features_train_1 = np.round(features_train_1.astype(float), decimals=4)
    features_train_1[np.isnan(features_train_1)] = 0

    features_test_1 = features_test_1.to_numpy()
    features_test_1 = np.round(features_test_1.astype(float), decimals=4)
    features_test_1[np.isnan(features_test_1)] = 0

    features_train_2 = np.round(features_train_2.astype(float), decimals=4)
    features_train_2[np.isnan(features_train_2)] = 0
    features_test_2 = features_test_2.to_numpy()

    features_test_2 = np.round(features_test_2.astype(float), decimals=4)
    features_test_2[np.isnan(features_test_2)] = 0

    if murmurs_1_nan_index != None:
        murmurs_train[np.isnan(murmurs_train)] = murmurs_1_nan_index
        murmurs_test[np.isnan(murmurs_test)] = murmurs_1_nan_index

    if outcomes_nan_index != None:
        outcomes_train[np.isnan(outcomes_train)] = outcomes_nan_index
        outcomes_test[np.isnan(outcomes_test)] = outcomes_nan_index

    return features_train_1, features_test_1, features_train_2, features_test_2,\
                murmurs_train, murmurs_test, outcomes_train, outcomes_test
