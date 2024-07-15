import numpy as np
from sklearn.metrics import classification_report

class MLP:
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr
         
    def forward(self, X):
        xs = [X]
        x = X
        for l in self.layers:
            xs.append(l.forward(x))
            x = xs[-1]
        return xs
    
    def train(self, X, y):
        xs = self.forward(X)
        y_pred = xs[-1]
        loss = cross_entropy(y, y_pred)
        sigma = gradient_loss(y, y_pred)
        w_next = None
        for l_idx in range(len(self.layers))[::-1]:
            layer = self.layers[l_idx]
            x_layer = xs[l_idx]
            w_save = layer.w.copy()
            sigma = layer.backprop(x_layer, sigma, w_next, self.lr)
            w_next = w_save
        return loss
    
    
    def predict(self, X):
        y_pred = self.forward(X)[-1]
        return y_pred.argmax(axis=-1)
    
    def batch_generate(self, X, y, batch_size):
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]

    def fit(self, X, y, X_test, y_test, batch_size = 8, epochs = 100):
        train_label = y.argmax(axis=1)
        test_label =y_test.argmax(axis=1)
        num_classes_train = len(np.unique(train_label))

        train_acc = []
        val_acc = []
        # f1_scores = []
        # recalls = []
        # precisions = []
        train_losses = []
        classification_reports = []
        
        for epoch in range(epochs):
            losses = []
            for train_x, train_y in self.batch_generate(X, y, batch_size):
                batch_loss = self.train(train_x, train_y)
                losses.append(batch_loss)

            train_pred = self.predict(X)
            val_pred = self.predict(X_test)
            
            train_acc.append((train_pred == train_label).mean())
            val_acc.append((val_pred == test_label).mean())
             
            classification_report_train = classification_report(train_label, train_pred, output_dict=True, zero_division=1)
            classification_report_val = classification_report(test_label, val_pred, output_dict=True, zero_division=1)
            classification_reports.append((classification_report_train, classification_report_val))
            train_losses.append(np.mean(losses))

        return train_acc, val_acc, classification_reports, train_losses


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        # -- Xavier initialization --
        # var = 2. / (input_size + output_size)
        # bound = np.sqrt(3.0 * var)
        # self.w = np.random.uniform(-bound, bound, size=(input_size, output_size))

        # -- He initilization --
        var = 2. / input_size
        bound = np.sqrt(3.0 * var)
        self.w = np.random.uniform(-bound, bound, size=(input_size, output_size))

        self.b = np.zeros(output_size)

    def grad_activation(self, x):
        if self.activation == 'sigmoid':
            a =  sigmoid(x)
            return a * (1 - a)
        if self.activation == 'relu':
            x_ = x.copy()
            x_[x_ <= 0] = 0
            x_[x_ > 0] = 1
            return x_
        
    def forward(self, X):
        if self.activation == "sigmoid":
            return sigmoid(np.dot(X, self.w) + self.b)
        elif self.activation == "None":
            return (np.dot(X, self.w) + self.b)
        elif self.activation == "relu":
            return relu(np.dot(X, self.w) + self.b)
        
    def backprop(self, X, sigma, w_next, lr):
        wx = np.dot(X, self.w) + self.b
        if w_next is not None:
            sigma = np.multiply(np.dot(sigma, w_next.T), self.grad_activation(wx))
        gradient_w = np.dot(X.T, sigma)
        gradient_b = sigma.sum(axis=0)
        self.w -= lr * gradient_w / X.shape[0]
        self.b -= lr * gradient_b / X.shape[0]
        return sigma



def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy(labels, logits):
    p = softmax(logits)
    loss = -np.mean(labels * np.log(p + 1e-15))
    return loss

def gradient_loss(labels, logits):
        p = softmax(logits)
        return p - labels