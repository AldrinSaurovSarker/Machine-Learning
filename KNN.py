# A KNN based machine learning model to predict the species of a flower

# Load the dataset and split into training and test dataset
def loadDataSet(filename, split, trainingSet=None, testSet=None):
    import csv  # To read dataset
    import random  # To randomly split dataset into training set and test set

    if testSet is None:
        testSet = []
    if trainingSet is None:
        trainingSet = []

    # Open file and convert into list
    with open(filename, 'r') as csvFile:
        lines = csv.reader(csvFile)
        dataSet = list(lines)

        for i in range(1, len(dataSet)):
            for j in range(len(dataSet[0])-1):
                dataSet[i][j] = float(dataSet[i][j])

            # Train-test split
            if random.random() < split:
                testSet.append(dataSet[i])
            else:
                trainingSet.append(dataSet[i])
        return dataSet


# Calculate distance between two points using euclidean formula
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += (float(instance1[i]) - float(instance2[i])) ** 2
    return distance ** 0.5


# Return labels of 'k' points from the training dataset which are closest to the test data point
def KNearestNeighbour(trainingSet, testData, k=5):
    distance = []  # Store distances of all training data point from test data point
    neighbours = []  # Store the closest 'k' neighbours
    length = len(trainingSet[0]) - 1

    for i in trainingSet:
        distance.append([euclideanDistance(i, testData, length), i])
    distance.sort()

    for i in range(k):
        neighbours.append(distance[i][1][-1])
    return neighbours


# Get the key of a value from a dictionary
def getKey(dic, val):
    for key, value in dic.items():
        if value == val:
            return key
    return 'Key Not Found'


# Get a list of labels and returns the label which occurred the most
def getLabel(neighbours):
    dic = {}

    # Creating a dictionary to count occurrences
    for i in neighbours:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1

    MaxValue = max(dic.values())
    return getKey(dic, MaxValue)


# Calculating the accuracy of the system
def accuracy_score(y_test, y_prediction):
    totalTest = len(y_prediction)
    correctTest = 0

    # Count total number of correct predictions
    for i in range(totalTest):
        if y_test[i][-1] == y_prediction[i]:
            correctTest += 1
    return 100 * correctTest / totalTest


# Taking 'k' = nearest lower odd number to the square root of total number of test samples
# Odd number is taken to avoid conflict
def KValue(num):
    val = int(num ** 0.5)
    return val - (1 - val % 2)


# The main function
# Split is the ratio of training dataset size : test dataset size
# By default, split is 0.2 so 80% of dataset will be training data and the rest 20% will be test data
def KNNBasedPredictionSystem(file_url, split=0.2):
    x_train = []
    x_test = []
    loadDataSet(file_url, split, x_train, x_test)
    k = KValue(len(x_test))

    y_pred = []  # To store the predicted outputs
    for x in x_test:
        neighbour = KNearestNeighbour(x_train, x, k)  # Get Neighbours
        y_pred.append(getLabel(neighbour))  # Predict

    print('Accuracy of the model : ',  accuracy_score(x_test, y_pred), '%')


url = 'Put the full url of Iris.csv'
KNNBasedPredictionSystem(url, 0.2)
