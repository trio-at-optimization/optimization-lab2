import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ==================================================================================================================== #

def print_dataset(dataset, X=None, y=None, y_pred=None, print_info=False, title=None):
    if X is None:
        X = dataset.data.values
    if y is None:
        y = dataset.target.to_numpy()
    if title is None and y_pred is None:
        title = "Dataset " + dataset.details['name']

    # ================================

    if print_info is True:
        print(dataset.DESCR)

    # ================================

    k_size_graphic = 4
    dot_size = 5 * k_size_graphic
    ncols = 5

    # --------------------------------
    n_features = X.shape[1]
    n_axs = n_features + 1 + 1
    nrows = math.ceil(n_axs / ncols)
    # --------------------------------

    width, height = ncols * k_size_graphic, nrows * k_size_graphic
    # width, height = 16, 10
    # ================================

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))

    if title != "":
        fig.suptitle(title)

    for feature_index, ax in enumerate(axs.flatten()):
        if feature_index < n_features:
            ax.scatter(X[:, feature_index], y, color='red', s=dot_size)
            if y_pred is not None:
                ax.scatter(X[:, feature_index], y_pred, color='blue', s=dot_size)
            ax.set_xlabel(dataset.feature_names[feature_index])
            ax.set_ylabel(dataset.target.name)
        else:
            fig.delaxes(ax)

    if nrows == 1:
        last_ax = axs[-1]
    else:
        last_ax = axs[-1, -1]

    if last_ax not in fig.axes:
        fig.add_axes(last_ax)
    last_ax.hist(y)
    last_ax.set_xlabel(dataset.target.name)
    last_ax.set_ylabel('Frequency')

    plt.scatter([], [], color='red', label='Real data', s=dot_size)
    if y_pred is not None:
        plt.scatter([], [], color='blue', label='Predictions', s=dot_size)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==================================================================================================================== #

def train_test_print_model(model, dataset, X=None, y=None, print_info=False, print_result=False, view_graphics=False,
                           title=None):
    if X is None:
        X = dataset.data.astype(float).values
    if y is None:
        y = dataset.target.astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if view_graphics:
        print_dataset(dataset, X_test, y_test, y_pred, print_info=print_info, title=title)
    if print_result:
        print('MSE:', mse)
        print('R^2:', r2)

    return mse, r2

# ==================================================================================================================== #
