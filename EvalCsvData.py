
from sklearn.metrics import accuracy_score
import CsvPreprocess as cpp
import MultiLayerPerceptron as mlp
  
features_train_1, features_test_1, features_train_2, features_test_2,\
    murmurs_train, murmurs_test, outcomes_train, outcomes_test = cpp.get_data()

hidden_layer_neurons = [32]
murmur_classifier = mlp.Mlp(hidden_layer_neurons,
                            features_train_1.shape[1], murmurs_train.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax,
                            verbose=True).fit(features_train_1.T, murmurs_train.T, batch_size=32)

outcome_classifier = mlp.Mlp(hidden_layer_neurons,
                            features_train_2.shape[1], outcomes_train.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax,
                            verbose=True).fit(features_train_2.T, outcomes_train.T, batch_size=32)

y_hat = murmur_classifier.predict_proba(features_test_1.T)
print("Murmurs accuracy: ", accuracy_score(murmurs_test, y_hat.T))

y_hat = murmur_classifier.predict_proba(features_test_2.T)
print("Murmurs accuracy: ", accuracy_score(outcomes_test, y_hat.T))
