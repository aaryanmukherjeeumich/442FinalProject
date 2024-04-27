import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeNormal

svm_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  
    'gamma': [0.01, 0.1], 
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # Kernel types
}

nn_params = {
    'kernel': ['HeNormal', 'uniform', 'normal'],
    'optimizer': ['adam', 'SGD', 'adagrad', 'adamax', 'nadam'],
    'batch_size': [8, 16, 32, 64, 128, 256, 512],
    'epochs': [5, 10, 15, 20, 30, 40, 50, 75, 100],
    'loss' : ['categorical_crossentropy']
}

def random_parameters(parameters_grid):
    selected_parameters = {}
    for param, values in parameters_grid.items():
        selected_parameters[param] = random.choice(values)
    return selected_parameters


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("-m", "--model", help = "Example: random_forest, svm, neural_net", required = False, default = "random_forest")
    #parser.add_argument("-a", "--autocorrect", help = "Example: on", required = False, default = "on")
    argument = parser.parse_args()

    return argument.model

def train_hand_model(model_type):
    f = None
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data = np.asarray(data_dict['data'])
    print("data shape:", data.shape)
    print("data[0]: ", data[0])
    # Print lengths of data elements
    for i, item in enumerate(data):
        if (len(item)) != 42:
            print("Length of element {} in data: {}".format(i, len(item)))
    labels = np.asarray(data_dict['labels'])
    print('labels type: ', labels.dtype)
    print("labels: ", labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = None
    if model_type=="random_forest":
        model = RandomForestClassifier()
        print("Running random forest classifier")
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)


    elif model_type=="svm":
        params = random_parameters(svm_params)
        print("Running support vector machine with parameters: ", params, "\n...\n")

        model = SVC(probability=True, **params)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        score = accuracy_score(y_predict, y_test)


    elif model_type=="neural_net":
        selected_parameters = random_parameters(nn_params)

        kern_init = selected_parameters['kernel']
        sel_optimizer = selected_parameters['optimizer']
        sel_batchsize = selected_parameters['batch_size']
        sel_epochs = selected_parameters['epochs']
        sel_loss = selected_parameters['loss']
        print("input shape???: ", x_train.shape[1])
        # Defining the model architecture
        if (kern_init == 'HeNormal'):
            classifier = Sequential([
            Input(shape=(x_train.shape[1],)),
            Dense(units=64, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=128, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=64, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=32, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=32, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=26, kernel_initializer=HeNormal())
        ])
        else: #use kern_init as value 'uniform' or 'normal'
            classifier = Sequential([
            Input(shape=(x_train.shape[1],)),
            Dense(units=64, kernel_initializer=kern_init, activation='relu'),
            Dense(units=128, kernel_initializer=kern_init, activation='relu'),
            Dense(units=64, kernel_initializer=kern_init, activation='relu'),
            Dense(units=32, kernel_initializer=kern_init, activation='relu'),
            Dense(units=32, kernel_initializer=kern_init, activation='relu'),
            Dense(units=26, kernel_initializer=kern_init)
        ])

        print("Running neural network with parameters: ", selected_parameters, "\n...\n")
        model = classifier
        model.compile(optimizer=sel_optimizer, loss=sel_loss, metrics=['accuracy'])
        print('compiled')
        # Training the model
        y_train_encoded = to_categorical(y_train, num_classes=26)
        y_test_encoded = to_categorical(y_test, num_classes=26)
        model.fit(x_train, y_train_encoded, batch_size=sel_batchsize, epochs=sel_epochs, verbose=0)
        print('fit')

        scores = model.evaluate(x_test, y_test_encoded)
        score = scores[1]
        
    

    print('{}% of samples were classified correctly !'.format(score * 100))

    f = open('model.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()

if __name__ == "__main__":
    model_type = parse_command_line_arguments()
    train_hand_model(model_type)
    

