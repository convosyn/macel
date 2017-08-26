from src import Neural
import pandas as pd
import numpy as np
import sys

USAGE = "Usage: python {!s} <in_file>"

def main(fin:str, learning_rate, ratio):
    print("loading data ...")
    data = pd.read_csv(fin)
    data = np.array(data)
    X = data[:, 1:]
    y = data[:, 0].reshape((-1, 1))
    clf = Neural.Neural(alpha = learning_rate, hidden_layer_size = 70, max_iters=50)
    m = X.shape[0]
    m_train = int(m * ratio)
    X_train = X[:m_train, :]
    y_train = y[:m_train, :]
    clf.fit(X_train, y_train, perform_gradient_checking = True, normalize =  True, show_cost=True)
    X_test = X[m_train:, :]
    y_test = y[m_train:, :]
    _, yp = clf.predict(X_train)
    yp = yp.reshape(-1, 1)
    #df = pd.DataFrame(yp, index=['op'])
    np.savetxt("fin_out", np.column_stack([yp, y_train]))
    #df.to_csv("fin_out", index = None, headers=None)
    print("Accuracy: {!s}".format(test_accuracy(y_train, yp)))

def test_accuracy(y_known, y_pred):
	return float(np.mean(y_known.reshape((-1, 1)) == y_pred.reshape((-1, 1)))) * 100


if __name__ == "__main__":
	lr = 1e-3
	ratio = 0.8
	if len(sys.argv) < 2:
		print(USAGE.format(sys.argv[0]))
		sys.exit(0)

	if len(sys.argv) == 3:
		lr = float(sys.argv[2])

	if len(sys.argv) == 4:
		ratio = float(sys.argv[3])
	main(sys.argv[1], lr, ratio)