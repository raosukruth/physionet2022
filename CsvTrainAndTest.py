from sklearn.metrics import accuracy_score
import CsvPreprocess as cpp
import MultiLayerPerceptron as mlp
import numpy as np

np.random.seed(2023)

mci_train, mci_test, oci_train, oci_test,\
    murmurs_train, murmurs_test, outcomes_train, outcomes_test = cpp.get_data()

assert np.isnan(murmurs_train).any() == False

hidden_layer_neurons = [14]
murmur_classifier = mlp.Mlp(hidden_layer_neurons,
                            mci_train.shape[1], murmurs_train.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax, lr=0.01,
                            verbose=True).fit(mci_train.T, murmurs_train.T, 
                                              batch_size=256, epochs=500)

y_hat = murmur_classifier.predict(mci_test.T)
assert (np.isnan(y_hat) == True).sum() == 0
print("Murmurs accuracy: ", accuracy_score(murmurs_test, y_hat.T))

hidden_layer_neurons = [14]
outcome_classifier = mlp.Mlp(hidden_layer_neurons,
                            oci_train.shape[1], outcomes_train.shape[1],
                            mlp.tanh, mlp.d_tanh, mlp.softmax, lr=0.013,
                            random_state=2023,
                            verbose=True).fit(oci_train.T, outcomes_train.T,
                                              batch_size=220, epochs=2200)

y_hat = outcome_classifier.predict(oci_test.T)
assert (np.isnan(y_hat) == True).sum() == 0
print("Outcomes accuracy: ", accuracy_score(outcomes_test, y_hat.T))
