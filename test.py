import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import pandas as pd
from sklearn.metrics import f1_score

np.random.seed(123)


class ConvNet:
    def __init__(self, no_filters_layer, filter_widths, X_tr, Y_tr, X_val, Y_val, d, K, n_len):
        self.X_train, self.X_val = X_tr, X_val
        self.Y_train, self.Y_val = Y_tr, Y_val
        self.d = d
        self.K = K

        # Hyper-parameters
        self.no_filters_layer = no_filters_layer  # Tuple of number of filters at layer index
        self.no_layers = len(self.no_filters_layer)
        self.filter_widths = filter_widths  # Tuple of number of filter widths at layer index
        self.filters = []  # Store filter matrices
        self.W = []
        self.n_len = np.zeros(self.no_layers+1)
        self.n_len[0] = n_len

        # Initialize filters, W and n_len
        self.initialize_cnn()

    def initialize_cnn(self):
        d = self.d
        k = self.filter_widths
        n = self.no_filters_layer

        for i in range(1, self.no_layers+1):
            self.n_len[i] = self.n_len[i - 1] - self.filter_widths[i-1] + 1

        # W has size: number of classes x filter size in final layer
        sig_W = np.sqrt(2 / n[-1])
        self.W = np.random.normal(0, sig_W, (self.K, int(self.n_len[-1] * n[-2])))

        self.filters.append(np.random.normal(0, np.sqrt(2 / (d + n[0])), (d, k[0], n[0])))

        for i in range(1, self.no_layers):
            sig_i = np.sqrt(2 / n[i - 1])
            self.filters.append(np.random.normal(0, sig_i, (n[i - 1], k[i], n[i])))

    def evaluate(self, X_batch):
        x = X_batch
        x_record = [x]

        for i in range(self.no_layers):
            MF = self.make_MF_matrix(self.n_len[i], self.filters[i])
            x = np.maximum(0, MF @ x)
            x_record.append(x)

        s = self.W @ x

        return self.softmax(s), x_record

    def compute_cost(self, X, Y):
        p, _ = self.evaluate(X)
        loss = - (1 / X.shape[1]) * np.sum(Y * np.log(p))

        return loss

    def compute_accuracy(self, X, Y):
        p, _ = self.evaluate(X)
        y = np.argmax(Y, axis=0)
        predicted = np.argmax(p, axis=0)
        acc = (predicted == y).mean()

        return acc

    def compute_gradients(self, X_batch, Y_batch):
        p_batch, x_record = self.evaluate(X_batch)
        g_batch = -(Y_batch - p_batch)

        n = x_record[-1].shape[1]
        grad_W = (1 / n) * g_batch @ x_record[-1].T
        grad_F = []

        g_batch = self.W.T @ g_batch
        g_batch = np.multiply(g_batch, np.where(x_record[-1] > 0, 1, 0))

        n = X_batch.shape[1]

        for i in range(len(x_record)-2, -1, -1):  # Start at the end, exclude the last & first batch
            grad = np.zeros(self.filters[i].shape)
            x = x_record[i]
            for j in range(n):
                g_j = g_batch[:, j]
                x_j = x[:, j]

                d, k, nf = self.filters[i].shape[0], self.filters[i].shape[1], self.filters[i].shape[2]
                MX = self.make_MX_matrix(x_j, d, k, nf, int(self.n_len[i]))
                v = g_j.T @ MX
                grad += (1 / n) * v.reshape(grad.shape, order='F')

            grad_F.append(grad)

            if i != 0:
                MF = self.make_MF_matrix(self.n_len[i], self.filters[i])
                g_batch = MF.T @ g_batch
                g_batch = np.multiply(g_batch, np.where(x > 0, 1, 0))

        temp = np.asarray(list(reversed(grad_F)))
        return temp, grad_W

    def compute_grad_num_slow(self, X_batch, Y_batch, h=1e-5):
        # Initialize all gradients to zero
        grad_W = np.zeros(self.W.shape)
        grad_F = []

        for i in range(len(self.filters)):
            grad_F.append(np.zeros(self.filters[i].shape))

        for j in range(self.W.shape[0]):
            for k in range(self.W.shape[1]):
                self.W[j, k] -= h
                c1 = self.compute_cost(X_batch, Y_batch)
                self.W[j, k] += 2 * h
                c2 = self.compute_cost(X_batch, Y_batch)
                self.W[j, k] -= h
                grad_W[j, k] = (c2 - c1) / (2 * h)

        for n in range(len(self.filters)-1, -1, -1):
            for j in range(self.filters[n].shape[0]):
                for k in range(self.filters[n].shape[1]):
                    for i in range(self.filters[n].shape[2]):
                        self.filters[n][j, k, i] -= h
                        c1 = self.compute_cost(X_batch, Y_batch)

                        self.filters[n][j, k, i] += 2 * h
                        c2 = self.compute_cost(X_batch, Y_batch)

                        self.filters[n][j, k, i] -= h
                        grad_F[n][j, k, i] = (c2 - c1) / (2 * h)

        return grad_F, grad_W

    def mini_batch_gd(self, n_batch, eta, rho, n_epochs, plot=False, balanced=False):
        n_step = 10
        train_costs = []
        val_costs = []

        train_accuracy = []
        val_accuracy = []

        prev_grad_W = 0
        prev_grad_F = [0 for _ in range(self.no_layers)]

        step = 0
        curr_best = (self.W, self.filters)

        for _ in tqdm(range(n_epochs)):
            if balanced:
                X_train, Y_train = self.upsample(self.X_train, self.Y_train)
            else:
                X_train, Y_train = self.X_train, self.Y_train

            for i in range(int(X_train.shape[1] / n_batch)):
                step += 1
                i_start = i * n_batch
                i_end = (i + 1) * n_batch

                X_batch = X_train[:, i_start:i_end]
                Y_batch = Y_train[:, i_start:i_end]
                grad_F, grad_W = self.compute_gradients(X_batch, Y_batch)
                self.W -= rho * prev_grad_W + eta * grad_W

                for k in range(len(self.filters)):
                    self.filters[k] -= rho * prev_grad_F[k] + eta * grad_F[k]

                if step % n_step == 0:
                    train_costs.append(self.compute_cost(self.X_train, self.Y_train))
                    train_accuracy.append(self.compute_accuracy(self.X_train, self.Y_train))

                    val_costs.append(self.compute_cost(self.X_val, self.Y_val))
                    val_accuracy.append(self.compute_accuracy(self.X_val, self.Y_val))

                    if len(val_costs) > 1 and val_costs[-1] < val_costs[-2]:
                        curr_best = (self.W, self.filters)
                    print("Step " + str(step) + ": " + str(min(val_costs)), min(train_costs))
                    print(max(val_accuracy))

                prev_grad_F = np.copy(grad_F)
                prev_grad_W = np.copy(grad_W)

        if plot:
            plt.plot([n_step*i for i in range(len(train_costs))], train_costs, label='Training Cost')
            plt.plot([n_step*i for i in range(len(val_costs))], val_costs, label='Validation Cost')
            plt.xlabel("Update Step")
            plt.ylabel("Cross Entropy Loss")
            plt.legend()
            plt.title("Training and Validation Cost")
            plt.show()

        print(max(val_accuracy), max(train_accuracy))
        self.W = curr_best[0]
        self.filters = curr_best[1]

    @staticmethod
    def make_MF_matrix(n_len, filters):
        dd, k, nf = filters.shape[0], filters.shape[1], filters.shape[2]

        MF = np.zeros((int((n_len - k + 1) * nf), int(n_len * dd)))
        VF = filters.reshape((dd * k, nf), order='F').T

        for i in range(int(n_len - k + 1)):
            MF[i * nf:(i + 1) * nf, dd * i:dd * i + dd * k] = VF

        return MF

    @staticmethod
    def make_MX_matrix(x_input, d, k, nf, nlen):
        if len(x_input.shape) > 1:
            x_input = x_input.flatten()

        MX_Matrix = np.zeros(((nlen - k + 1) * nf, k * nf * d))

        for i in range(nlen - k + 1):
            MX_Matrix[i * nf: i * nf + nf, :] = block_diag(*[x_input[d * i: d * i + k * d] for _ in range(nf)])

        return MX_Matrix

    @staticmethod
    def softmax(s):
        return np.exp(s - np.max(s, axis=0)) / np.exp(s - np.max(s, axis=0)).sum(axis=0)

    def predict(self, X):
        p, _ = self.evaluate(X)
        return np.argmax(p, axis=0)

    def compute_class_accuracy(self, X, Y):
        y = np.argmax(Y, axis=0)
        pred = self.predict(X)

        data_per_class = dict((x, list(pred).count(x)) for x in set(list(pred)))
        accuracy = dict((label, 0) for label in sorted(set(pred)))

        for pair in zip(y, pred):
            if pair[0] == pair[1]:
                accuracy[pair[0]] += 1

        return {k: 100*float(accuracy[k])/data_per_class[k] for k in accuracy}

    @staticmethod
    def upsample(X, Y):
        labels = np.argmax(Y, axis=0)
        data_per_class = dict((x, list(labels).count(x)) for x in set(list(labels)))
        min_label = min(data_per_class.values())
        indices = np.asarray([])

        for label in set(labels):
            subset = np.where(labels == label)
            idx = np.random.choice(subset[0], size=min_label, replace=False)
            indices = np.concatenate([indices, idx])

        np.random.shuffle(indices)
        indices = indices.astype(int)

        return X[:, indices], Y[:, indices]


class DataLoader:
    def __init__(self):
        self.name_path = "C:/Users/g022191/PycharmProjects/DD2424/Lab_3/Data/ascii_names.txt"
        self.names, self.labels = self.read_names_and_labels(self.name_path)
        self.n_len = -1
        self.d = 0
        self.K = 0

    @staticmethod
    def read_names_and_labels(file_path):
        names = []
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                split_fields = line.split(" ")
                names.append(' '.join(split_fields[:-1]))  # Append name
                labels.append(split_fields[-1])  # Append label

        return names, labels

    @staticmethod
    def encode_string(string, character_dictionary, max_length):
        d = len(character_dictionary)
        encoded_string = np.zeros((d, max_length))
        for i in range(len(string)):
            encoded_string[character_dictionary[string[i]], i] = 1
        return encoded_string

    @staticmethod
    def one_hot_encoding(label_id, label_no):
        vector = np.zeros(label_no)
        vector[label_id] = 1
        return vector

    def get_dataset(self):
        self.n_len = -1  # Maximum name length
        char_idx = 0
        character_dict = {}

        for name in self.names:
            # Compare current length with maximum name length
            cur_len = len(name)
            if cur_len > self.n_len:
                self.n_len = cur_len
            # Store any previously unseen characters into dictionary
            for character in name:
                if character not in character_dict.keys():
                    character_dict[character] = char_idx
                    char_idx += 1
        labels = np.array(self.labels, dtype=int)
        self.d = len(character_dict)  # number of unique characters
        self.K = len(np.unique(labels))  # number of unique classes

        vectorized_input_size = self.d * self.n_len
        X = np.zeros((vectorized_input_size, len(self.names)))
        for idx, name in enumerate(self.names):
            X[:, idx] = self.encode_string(name, character_dict, self.n_len).flatten(order='F')

        test_names, test_labels = self.read_names_and_labels("C:/Users/g022191/PycharmProjects/DD2424/Lab_3/Data/test_names")
        X_test = np.zeros((vectorized_input_size, len(test_names)))
        for idx, name in enumerate(test_names):
            X_test[:, idx] = self.encode_string(name, character_dict, self.n_len).flatten(order='F')

        # One-hot encoding for output
        Y = np.array([self.one_hot_encoding(label - 1, self.K) for label in labels]).T

        return X, Y, X_test


def getRelativeErrors(grads_numerical, grads_analytical, eps=1e-6):
    rel_errors = []
    for grad_n, grad_a in zip(grads_numerical, grads_analytical):
        abs_diff = np.absolute(grad_n - grad_a)
        abs_sum = np.absolute(grad_n) + np.absolute(grad_a)
        max_elems = np.where(abs_sum > eps, abs_sum, np.finfo(float).eps)
        rel_errors.append(abs_diff / max_elems)

    return rel_errors


def check_gradients(CNN, X, Y):
    grad_F_n, grad_W_n = CNN.compute_grad_num_slow(X[:, :10], Y[:, :10])
    grad_F_a, grad_W_a = CNN.compute_gradients(X[:, :10], Y[:, :10])

    errors_F = getRelativeErrors(grad_F_n, grad_F_a)
    errors_W = getRelativeErrors(grad_W_n, grad_W_a)

    for i, el in enumerate(errors_F):
        print('Max Error for F' + str(i) + ":", np.max(el))

    print('Max Error for W:', np.max(errors_W))


def main():
    data_loader = DataLoader()
    X, Y, X_test = data_loader.get_dataset()

    # Get the indices of the inputs that are going to used in the validation set
    val_ind_path = "C:/Users/g022191/PycharmProjects/DD2424/Lab_3/Data/Validation_Inds.txt"
    validation_indices = np.loadtxt(val_ind_path, dtype=int)

    # Split data into training and validation sets
    X_tr = np.delete(X, validation_indices, axis=1)
    X_val = X[:, validation_indices]
    Y_tr = np.delete(Y, validation_indices, axis=1)
    Y_val = Y[:, validation_indices]

    # Shuffle data
    indices = np.arange(0, X_tr.shape[1] - 1, 1)
    np.random.shuffle(indices)
    X_tr, Y_tr = X_tr[:, indices], Y_tr[:, indices]

    # Check distribution of classes
    data_per_class = dict((x, list(np.argmax(Y_val, axis=0)).count(x)) for x in set(list(np.argmax(Y_val, axis=0))))
    factor = 100.0 / sum(data_per_class.values())
    for k in data_per_class:
        data_per_class[k] = round(data_per_class[k] * factor, 2)

    print(data_per_class)

    # Train CNN
    CNN = ConvNet(no_filters_layer=(50, 50),
                  filter_widths=(5, 3),
                  X_tr=X_tr,
                  Y_tr=Y_tr,
                  X_val=X_val,
                  Y_val=Y_val,
                  d=data_loader.d,
                  K=data_loader.K,
                  n_len=data_loader.n_len)

    CNN.mini_batch_gd(n_batch=10,
                      n_epochs=10,  # 20 000 steps = 2223 for balanced, 117 for imbalanced
                      eta=0.001,
                      rho=0.9,
                      plot=True,
                      balanced=True)

    # Evaluate CNN
    y = pd.Series(np.argmax(Y_val, axis=0), name="Actual")
    pred = pd.Series(CNN.predict(X_val), name="pred")
    df_confusion = pd.crosstab(y, pred)

    print(CNN.compute_class_accuracy(X_val, Y_val))
    print(df_confusion)
    print(f1_score(y, pred, average='weighted'))


if __name__ == '__main__':
    main()
