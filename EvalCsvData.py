
from sklearn.metrics import accuracy_score
import CsvPreprocess as cpp
import MultiLayerPerceptron as mlp
import numpy as np
  
mci_train, mci_test, oci_train, oci_test,\
    murmurs_train, murmurs_test, outcomes_train, outcomes_test = cpp.get_data()

assert np.isnan(murmurs_train).any() == False

def scale_features(f):
    print("f shape", f.shape)
    fmean = np.mean(f, axis=1, keepdims=True)
    fminmax = np.max(f, axis=1, keepdims=True) - np.min(f, axis=1, keepdims=True)
    f = (f - fmean) / fminmax
    assert np.isnan(f).any() == False, "f has nans {}".format(f)
    return f

# Scale features
mci_train = scale_features(mci_train)
oci_train = scale_features(oci_train)
mci_test = scale_features(mci_test)
oci_test = scale_features(oci_test)

hidden_layer_neurons = [32]
murmur_classifier = mlp.Mlp(hidden_layer_neurons,
                            mci_train.shape[1], murmurs_train.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax,
                            verbose=True).fit(mci_train.T, murmurs_train.T, epochs=2000)

y_hat = murmur_classifier.predict(mci_test.T)
assert (np.isnan(y_hat) == True).sum() == 0
print("Murmurs accuracy: ", accuracy_score(murmurs_test, y_hat.T))

outcome_classifier = mlp.Mlp(hidden_layer_neurons,
                            oci_train.shape[1], outcomes_train.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax,
                            verbose=True).fit(oci_train.T, outcomes_train.T, batch_size=2000)

y_hat = outcome_classifier.predict(oci_test.T)
assert (np.isnan(y_hat) == True).sum() == 0
print("Outcomes accuracy: ", accuracy_score(outcomes_test, y_hat.T))
