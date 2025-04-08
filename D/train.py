import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_features(text):
    feature_list = []
    rows = text.split("\n")
    for row in rows:
        row = row.strip().split()
        if not row:
            continue
        feature = list(map(float, row))
        feature_list.append(feature)
    feature_list = np.array(feature_list)
    return feature_list


def parse_labels(text):
    labels = list(map(int, labels_txt.split()))
    labels = np.array(labels)
    return labels


def main(feature_txt, labels_txt):
    # DATA READ
    X = parse_features(feature_txt) #(n_samples, n_features)
    y = parse_labels(labels_txt) #(n_samples, )
    
    # DATA SPLIT
    N = len(X)
    train_size = N // 2
    test_size = N - train_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # TRAIN
    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)

    # TEST
    y_predictions = model.predict(X_test)
    return accuracy_score(y_test, y_predictions)


feature_txt = open("output.txt", mode="r").read()
labels_txt = open("labels.txt", mode="r").read()

accuracy = main(feature_txt, labels_txt)

print("ACCURACY:", accuracy * 100)

