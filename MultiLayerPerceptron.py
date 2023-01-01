import numpy as np
from scipy.special import xlogy

def relu(z):
    return np.maximum(z, 0)

def d_relu(z):
    return np.maximum(z > 0, 0)

def softmax(z):
    # To avoid overflow, reduce the exponent
    low_z = z - np.max(z)
    result = np.exp(low_z) / np.sum(np.exp(low_z), axis=0, keepdims=True)
    return result

def cross_entropy_loss(y_hat, y):
    return -xlogy(y, y_hat).sum() / y_hat.shape[0]  # scikit-learn implementation

def d_cross_entropy_loss(y_hat, y):
    return y_hat - y

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_fn, gradient_fn):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_fn = activation_fn
        self.gradient_fn = gradient_fn
        
        self.w = np.random.randn(num_neurons, num_inputs)
        self.b = np.random.randn(num_neurons, 1)
        self.mw = np.zeros(self.w.shape)
        self.mb = np.zeros(self.b.shape)
        self.vw = np.zeros(self.w.shape)
        self.vb = np.zeros(self.b.shape)
        self.A = None
        self.Z = None
        
    def output(self, x):
        self.Z = np.dot(self.w, x) + self.b
        self.A = self.activation_fn(self.Z)
        return self.A
    
class Mlp:
    def __init__(self, hidden_neurons, num_features, num_outputs, 
                 hidden_activation_fn, hidden_gradient_fn, 
                 output_activation_fn, lr=0.01, verbose=False):
        np.random.seed(42)
        self.layers = []
        num_inputs = num_features
        for neurons in hidden_neurons:
            self.layers.append(Layer(neurons, num_inputs, 
                                     hidden_activation_fn, 
                                    hidden_gradient_fn))
            num_inputs = neurons
        
        # Add the output layer. Number of neurons in the output layer 
        # is num_outputs and number of inputs is number of neurons of the 
        # last hidden layer
        self.layers.append(Layer(num_outputs, num_inputs,
                                 output_activation_fn, None))
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.output_activation_fn = output_activation_fn
        self.lr = lr
        self.verbose = verbose
        self.data_store = {}
  
    def forward(self, x):
        inp = x
        output = None
        for layer in self.layers:
            output = layer.output(inp)
            inp = output
        return output
    
    def predict_proba(self, x):
        return self.forward(x)
    
    def predict(self, x):
        y_hat = np.zeros([self.num_outputs, x.shape[1]]).astype(int)
        output = self.predict_proba(x)
        y_hat[np.argmax(output, axis=0), np.arange(output.shape[1])] = 1
        return y_hat
    
    def backward(self, x, y, y_hat, beta1=0.9, beta2=0.999, epsilon=1e-8):
        num_samples = x.shape[1]
        dz =  d_cross_entropy_loss(y_hat, y)
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            prev_layer = self.layers[idx - 1] if idx else None
            
            inp = x if idx == 0 else prev_layer.A
            dw = np.dot(dz, inp.T) / num_samples
            db = np.sum(dz, axis=1, keepdims=True) / num_samples
            if idx:
                dz = np.multiply(np.dot(layer.w.T, dz), 
                             prev_layer.gradient_fn(prev_layer.A))
        
            layer.mw = beta1 * layer.mw + (1 - beta1) * dw
            layer.mb = beta1 * layer.mb + (1 - beta1) * db
            
            layer.vw = beta2 * layer.vw + (1 - beta2) * (dw ** 2)
            layer.vb = beta2 * layer.vb + (1 - beta2) * (db ** 2)
            
            # update weights and bias
            layer.w -= (self.lr * (layer.mw / (np.sqrt(layer.vw) + epsilon)))
            layer.b -= (self.lr * (layer.mb / (np.sqrt(layer.vb) + epsilon)))
            
    def make_mini_batches(self, x, y, batch_size):
        batches = []
        for b in range(0, x.shape[1], batch_size):
            x_batch = x[:, b:b + batch_size]
            y_batch = y[:, b:b + batch_size]
            batches.append((x_batch, y_batch))
        return batches
                             
    def fit(self, x, y, batch_size=200, epochs=500):
        for e in range(epochs):
            cost = 0
            batches = self.make_mini_batches(x, y, batch_size)
            for x_batch, y_batch in batches:    
                y_hat = self.forward(x_batch)
                cost = cross_entropy_loss(y_hat, y_batch)
                self.backward(x_batch, y_batch, y_hat)
            if self.verbose and (e % 500 == 0):
                print('cost after epoch {} is {}'.format(e, cost))
        return self
