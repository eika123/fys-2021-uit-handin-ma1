import numpy as np 
import pandas as pd 
from IPython.display import display
from unzip_dataset import extract_dataset, DATASET_FOLDER_NAME

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# for plotting contour line for classification.
# see https://stackoverflow.com/a/17432641
import matplotlib.cm as cm

from collections.abc import Callable

# from collections.abc import Tuple

def get_data(training:bool = True, normalize=True, splitpercent=0.8) -> dict[str: np.ndarray]:
    extract_dataset()

    datasetname = f'{DATASET_FOLDER_NAME}/SpotifyFeatures.csv'

    dataset = pd.read_csv(datasetname)
    #
    # https://www.geeksforgeeks.org/display-the-pandas-dataframe-in-table-style/
    # Also see for the Ipython.display import above
    # display(dataset)
    
    #dataset = dataset.query('genre=="Pop" | genre=="Classical"')
    dataset = dataset.query('genre=="Pop" | genre=="Classical"')

    number_of_popsongs = dataset.query('genre=="Pop"').shape[0]
    number_of_classicals = dataset.query('genre=="Classical"').shape[0]

    print(f"number of Pop songs:       {number_of_popsongs}")
    print(f"number of Classical songs: {number_of_classicals}")

    

    def labeler(genre):
        if genre == 'Pop':
            return 1
        else:
            return 0

    vlabeler = np.vectorize(labeler)


    dataset['labels'] = vlabeler(dataset['genre'])

    if training:

       # find indexes belonging to each label
        pop_indices = np.array(dataset.groupby(by="labels").indices[1])
        classic_indices = np.array(dataset.groupby(by="labels").indices[0])

        # shuffle at random 
        rng = np.random.default_rng()
        rng.shuffle(pop_indices)
        rng.shuffle(classic_indices)

        # split the data
        sp = splitpercent
        training_indices = np.concatenate((pop_indices[:int(sp*number_of_popsongs)], classic_indices[:int(sp*number_of_classicals)]))
        testing_indices = np.concatenate((pop_indices[int(sp*number_of_popsongs):], classic_indices[int(sp*number_of_classicals):]))

        rng.shuffle(training_indices)
        rng.shuffle(testing_indices)

        liveness = np.array(dataset['liveness'])
        loudness = np.array(dataset['loudness'])
        labels = np.array(dataset['labels'])

        if normalize:
            liveness /= np.max(np.max(abs(liveness)))
            loudness /= np.max(np.max(abs(loudness)))

        features_training = np.matrix([ liveness[training_indices], loudness[training_indices] ]).transpose()
        labels_training = np.matrix(labels[training_indices]).transpose()

        print(f"features_training.shape = {features_training.shape}")
        print(f"labels_training.shape = {labels_training.shape}")

        features_testing = np.matrix([ liveness[testing_indices], loudness[testing_indices] ]).transpose()
        labels_testing = np.matrix(labels[testing_indices]).transpose()

        return {'training-features': features_training, 'training-labels': labels_training, 'testing-features': features_testing, 'testing-labels': labels_testing}
    
    else:
        features = np.matrix([dataset['liveness'], dataset['loudness']]).transpose()
        labels = np.matrix(dataset['labels']).transpose()

        if normalize:
            features /= np.max(np.max(abs(features)))

        return {'features': features, 'labels': labels}


def visualize_data(dataset: dict[str: np.ndarray], title: str = None) -> None:
    """
    dataset must be a dictionary with keys 'features' and 'labels' 
    features can be an nx2 array, while the labels must be nx1.
    """
    X, Y = np.array(dataset['features'].transpose()[0]), np.array(dataset['features'].transpose()[1])
    colors = np.array(dataset['labels'])

    plt.figure()
    plt.scatter(X, Y, c=colors)
    plt.xlabel('liveness')
    plt.ylabel('loudness')
    if title:
        plt.title(title)
    plt.show()


def grad_loss_gd(X: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    X     - features 
    y_hat - predicted values
    y     - labels
    """
    #Xt = X.transpose()
    Xt = X.T
    return (Xt @ (y_hat - y)) 


def grad_loss_sgd(Xi: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return Xi*(y_hat - y)
    

def loss_function(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    column vectors Nx1 u and y.
    N is number of samples 
    """
    log = np.log
    yt = y.T

    return ( -np.sum(yt*log(u) + (1 - yt)*log(1 - u)) )


def sigmoid(x):
    max_x = 1e2 # hack
    eps = 1e-4
    return np.exp(np.minimum(x, max_x))/(1 + np.exp(np.minimum(x, max_x)))*(1 - eps) + eps/2


def logreg_model_gd(a: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    a   -   model parameters  -- column vector           N x (f + 1)        f is number of features
    X   -   features          -- feature matrix          N x f

    N is the number of samples
    f is the number of features

    returns predicted values
    """
    # print(f"logreg_model: np.max(X@a) = {np.max(X@a)}")
    r = sigmoid(X@a)
    return r

def logreg_model_sgd(weights:np.ndarray, features: np.ndarray) -> float:
    # dot product
    return sigmoid(weights @ features)

def learn_gradient_descent(trainingset: dict[str, np.ndarray],
                                model: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                L: Callable[[np.ndarray, np.ndarray], float],
                                gradL: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                                learning_rate:float = 0.0001,
                                parameter_dim:int|None = None,
                                initial_parameters:None|np.ndarray = None) -> dict[str: np.ndarray]:
    """
    model: the parametric model 
    L: the loss function    -    we minimize the loss function L(u, y) where labels are y and the
                                 predicted values are u with parameters a fed into the model
    gradL: the gradient of the loss function
    initial_parameters: you can make a guess
    """

    eps = 1e-4

    X = trainingset['features']
    y = trainingset['labels']

    number_of_samples = max(y.shape)
    print(f"number of samples = {number_of_samples}")

    bias = np.matrix(np.ones(max(y.shape))).transpose()
    print(f"bias.shape = {bias.shape}          X.shape = {X.shape}")

    X = np.concatenate((X, bias), axis=1)

    if not initial_parameters:
        weights = np.matrix(np.ones(X.shape[1])).transpose()
    else:
        weights = initial_parameters

    y_hat = model(weights, X) #prediction
    loss = L(y_hat, y)
    loss_track = []

    l1, l2 = abs(loss), abs(loss) + eps + 10

    i = 0
    while abs(l1 - l2) > eps:
        gradient = gradL(X, y_hat, y)
        weights = weights - gradient*learning_rate         # update parameters
        y_hat = model(weights, X)
        
        
        
        loss_track.append(l1)
        l2 = l1
        l1 = abs(L(y_hat, y))
        i += 1
        if i % 100 == 0:
            print(f"l1 = abs(L(u, y)) = {l1},                l2 = {l2},        abs(l1 - l2) = {abs(l1 - l2)}")
            i = 0
    
    
    return {'parameters': weights, 'loss_track': np.array(loss_track), 'features+bias': X}


def learn_stochastic_gradient_descent(trainingset: dict[str, np.ndarray],
                                     model: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     L: Callable[[np.ndarray, np.ndarray], float],
                                     gradL: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     learning_rate:float = 0.05,
                                     epochs = 20,
                                     initial_parameters:None|np.ndarray = None) -> dict[str: np.ndarray]:
    """
    model: the parametric model 
    L: the loss function    -    we minimize the loss function L(u, y) where labels are y and 
                                 the predicted values are u with parameters a fed into the model
    gradL: the gradient of the loss function
    initial_parameters: you can make a guess
    """

    X = trainingset['features']
    Y = trainingset['labels']
    number_of_samples = max(Y.shape)
    print(f"number of samples = {number_of_samples}")
    bias = np.matrix(np.ones(number_of_samples)).transpose()


    X = np.concatenate((X, bias), axis=1)

    if initial_parameters == None:
        #weights = np.matrix(np.ones(X.shape[1])).transpose()
        weights = np.ones(X.shape[1])
        #weights = np.array([-0.53479475, 15.85796746, 4.16622321])
        # weights = np.array([-1.47070523, 19.98291305, 5.26484238])
        #weights = np.array([-1.62023173, 20.83070842, 5.44901886])
    else:
        weights = initial_parameters

    # y_hat = model(weights, X)
    # loss = L(y_hat, Y)
    loss_track = np.zeros(number_of_samples)
    overall_loss_track = np.zeros(epochs)

    ## implements pseudo-code from lecture slides on stochastic gradient descent
    for e in range(epochs):
        for i in range(number_of_samples):

            #turn 1D-matrix into array: https://stackoverflow.com/a/3338368
            features = np.squeeze(np.asarray(X[i]))     # i'th row (sample) in the feature matrix
            y = Y[i,0]                                  # i'th label

            y_hat = model(weights, features)
            gradient = gradL(features, y_hat, y)
            weights = weights - gradient*learning_rate

            loss = L(y_hat, y)
            loss_track[i] = loss

            if (i + 1) % number_of_samples == 0:
                overall_loss_track[e] = np.sum(loss_track)
                print(f"weights = {weights}     loss = {overall_loss_track[e]}")
                
    return {'parameters': weights, 'loss_track': np.array(overall_loss_track), 'features+bias': X}


def visualize_model(weights, model, data):
    # scatterplot on top of contour plot https://stackoverflow.com/a/17432641
    # remember f: R² --> [0, 1]

    pass




def classifier(weights, model, sample):
    return 1 if model(weights, sample) >= 0.5 else 0


def evaluate_model(weights, model, features, labels):

    N = max(features.shape) # number of samples
    
    predictions = np.zeros(N)

    correct_predictions = 0
    for i in range(N):
        sample_i = np.squeeze(np.asarray(features[i]))  # i'th row in the feature matrix  -- turn matrix into 1-D-ndarray
        label_i = np.squeeze(np.asarray(labels[i]))
        # print(f"label_i = {label_i},      type(label_i) = {type(label_i)}")
        prediction = int(classifier(weights, model, sample_i))
        predictions[i] = prediction
        correct_pred = int(prediction == label_i)
        correct_predictions += correct_pred
    
    return (correct_predictions / N), predictions


def model_summary(results: dict[str: np.matrix | np.ndarray], model, dataset_training, dataset_testing):
    print()
    print("###################### model summary #######################################")
    weights, features_training, labels_training = results['parameters'], results['features+bias'], dataset_training['labels']

    print(f"weights = {weights}")

    number_of_samples = max(dataset_testing['labels'].shape)
    bias = np.matrix(np.ones(number_of_samples)).transpose()
    features_testing = dataset_testing['features']
    features_testing = np.concatenate((features_testing, bias), axis=1)
    labels_testing = dataset_testing['labels']

    accuracy_training_set, predictions_training_set = evaluate_model(weights, model, features_training, labels_training)
    accuracy_testing_set, predictions_testing_set = evaluate_model(weights, model, features_testing, labels_testing)
    print(f"accuracy on training set = {accuracy_training_set}")
    print(f"accuracy on test set = {accuracy_testing_set}")


    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    confm_training = confusion_matrix(np.asarray(labels_training), predictions_training_set)
    confm_testing = confusion_matrix(np.asarray(labels_testing), predictions_testing_set)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay 
    disp_tr = ConfusionMatrixDisplay.from_predictions(predictions_training_set, np.asarray(labels_training))
    disp_tr.plot()
    plt.title('Training set. 1 is pop music, 0 is classical')
    plt.show()

    disp_te = ConfusionMatrixDisplay.from_predictions(predictions_testing_set, np.asarray(labels_testing))
    disp_te.plot()
    plt.title('Testing set. 1 is pop music, 0 is classical')
    plt.show()





def evaluate_sgd(results: dict[str: np.matrix | np.ndarray], dataset_training, dataset_testing):
    model_summary(results, logreg_model_sgd, dataset_training, dataset_testing)

def evaluate_gd(results: dict[str: np.matrix | np.ndarray], dataset_traning, dataset_testing):
    model_summary(results, logreg_model_gd, dataset_training, dataset_testing)



if __name__ == '__main__':
    dataset_learning = get_data(splitpercent=0.8)
    dataset_training = {'features': dataset_learning['training-features'], 'labels': dataset_learning['training-labels']}
    dataset_testing = {'features': dataset_learning['testing-features'], 'labels': dataset_learning['testing-labels']}

    # visualize_data(get_data(training=False))
    visualize_data(dataset_training, title="Training set - normalized data")

    # print(np.max(dataset_training['features']))
    # print(dataset_training['features'])
    # results_gd = learn_gradient_descent(dataset_training,
    #                 logreg_model_gd,
    #                 loss_function,
    #                 grad_loss_gd,
    #                 learning_rate=0.0005
    #                 )

    # plt.figure()
    # plt.plot(results_gd['loss_track'][5:])
    # plt.yscale('log')
    # plt.show()

    # evaluate_gd(results_gd, dataset_training, dataset_testing)

    results_sgd = learn_stochastic_gradient_descent(dataset_training,
                    logreg_model_sgd,
                    loss_function,
                    grad_loss_sgd,
                    learning_rate=0.005,
                    epochs=10,
                    )

    plt.figure()
    plt.plot(results_sgd['loss_track'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('cross entropy loss')
    plt.yscale('log')
    plt.show()

    evaluate_sgd(results_sgd, dataset_training, dataset_testing)
    