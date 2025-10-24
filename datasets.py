import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

__all__ = ["make_sin_dataset", "make_diabetes_dataset"]


def make_sin_dataset(dataset_size=100, test_size=0.4, noise_value=0.4, plot=False):
    """1D regression problem y=sin(x*\pi)"""
    x_ax = np.linspace(-1, 1, dataset_size)
    y = [[np.sin(x * np.pi)] for x in x_ax]
    np.random.seed(123)
    noise = np.array([np.random.normal(0, 0.5, 1) for i in y]) * noise_value
    X = np.array(x_ax)
    y = np.array(y + noise)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=40, shuffle=True
    )

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    scaler = MinMaxScaler(
        feature_range=(-1, 1)
    )  # scaling in the range of quantum circuits
    y = scaler.fit_transform(y)
    y_test = scaler.transform(y_test)

    # reshaping for computation
    y_train = y_train.reshape(
        -1,
    )
    y_test = y_test.reshape(
        -1,
    )

    return X_train, X_test, y_train, y_test


from sklearn.datasets import load_diabetes


def make_diabetes_dataset(dataset):
    """Loading diabetes dataset and dropping 2 random elements to match
    the dataset size presented in arXiv:2410.19921"""
    ## dataset loading in Xy format
    X, y = load_diabetes(return_X_y=True)

    ## we need bmi (2) and s5 (8) as features
    selector = np.zeros(10, dtype=bool)
    selector[2], selector[8] = 1, 1

    X = X[:, selector]

    np.random.seed(123)
    drop_items = np.random.choice(range(442), 2)
    print("removed elements:", drop_items)

    X = np.delete(X, drop_items, axis=0)
    y = np.delete(y, drop_items)

    test_size = 400 / 440

    # split the dataset
    ### diabetes
    if dataset == "diabetes":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=52, shuffle=True
        )

    ## diabetes2
    elif dataset == "diabetes2":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=5, shuffle=True
        )

    scalerX = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = scalerX.fit_transform(X_train)
    X_test = scalerX.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    scalery = MinMaxScaler(
        feature_range=(-1, 1)
    )  # scaling in the range of quantum circuits
    y_train = scalery.fit_transform(y_train)
    y_test = scalery.transform(y_test)

    # reshaping for computation
    y_train = y_train.reshape(
        -1,
    )
    y_test = y_test.reshape(
        -1,
    )

    return X_train, X_test, y_train, y_test
