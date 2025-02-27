import pandas as pd
import numpy as np

def main():
    file = str(input("Welcome to my Feature Selection Algorithm!\nPlease type in the name of the file to test the algorithm (using format: name_of_file.txt): \n"))
    selection = input("\nNow, please type in the number of the feature selection algorithm you want to run:\n\n1) Forwards Selection\n2) Backwards Selection\n")
    df = pd.read_csv(file, delimiter=r"\s+", header=None, dtype=float)
    print("This dataset has " + str(df.shape[1]-1) + " columns, not including the class attribute. It has " + str(df.shape[0]) + " instances.")
    base_accuracy = get_accuracy(df)
    print("Running nearest neighbor with all " + str(df.shape[1]-1) + " features, using 'leave-one-out' evaluation, I get an accuracy of " + str(base_accuracy)) 

    forward_selection(df)

def forward_selection(df):
    highest_accuracy_features = []
    current_features = []
    max_accuracy = 0
    for i in range(1, df.shape[1]):
        print("On level " + str(i) + " of the search tree")
        local_max_feature = None
        local_max_accuracy = 0
        for j in range(1, df.shape[1]):
            if(j in current_features or j in highest_accuracy_features):
                continue
            print("Considering adding " + str(j) + " feature")
            accuracy = get_accuracy(df, current_features, j, 1)
            print(accuracy)
            if accuracy > local_max_accuracy:
                local_max_accuracy = accuracy
                local_max_feature = j
        # append the local maximum feature, even if it isn't the best accuracy -- ensures that there's no infinite loop by continuing the greedy algorithm
        current_features.append(local_max_feature)
        # if the local max is more than our actual max, then we should make that the set of features
        if local_max_accuracy < max_accuracy:
            print("The overall accuracy has decreased. I continue the search in case of a local maximum.")
            continue
        else:
            print("here")
            max_accuracy = local_max_accuracy
            highest_accuracy_features = current_features.copy()
            print(max_accuracy)
            print("On level " + str(i) + ", I added feature " + str(local_max_feature))
    print(highest_accuracy_features)

def backward_selection(df):
    current_features = []
    for i in range(1, df.shape[1]-1):
        print("On the " + str(i) + "th level of the search tree")
        feature = None
        max_accuracy = 0
        for j in range(1, df.shape[1]):
            if(j not in current_features):
                print("Considering adding " + str(j) + " feature")
                # accuracy = accuracy(df, current_features, j)
                # get_accuracy(df, current_features, j)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    feature = j
        current_features.append(feature)
        print("On level " + str(i) + ", I added feature " + str(feature))


def get_accuracy(df, current_features = None, feature = None, selection_type = 0):
    # using numpy since it is better for performing calculations
    # resources: https://stackoverflow.com/questions/52670767/summing-array-values-over-a-specific-range-with-numpy
        # https://numpy.org/doc/2.1/reference/generated/numpy.array.html
        # https://www.geeksforgeeks.org/python-numpy/
    data = df.to_numpy()
    labels = data[:, 0]
    if selection_type == 0: # all columns case
        selected_features = list(range(1, data.shape[1]))  
    elif selection_type == 1: # forward selection
        selected_features = list(current_features) + [feature] if feature not in current_features else list(current_features)
    else: #backward selection
        selected_features = [col for col in current_features if col != feature]
    
    # getting the features to check for accuracy
    features = data[:, selected_features]
    correctly_classified = 0
    num_samples = features.shape[0]

    # looping through all rows
    for i in range(num_samples):
        obj_to_classify = features[i] # row i
        distances = np.sqrt(np.sum((features - obj_to_classify) ** 2, axis=1)) # get distance from all other rows
        distances[i] = np.inf #remove case of self
        nn_index = np.argmin(distances)  # smallest distance = label that we want
        nn_label = labels[nn_index]
        if nn_label == labels[i]:
            correctly_classified += 1
    accuracy = correctly_classified / num_samples
    return accuracy

if __name__ == "__main__":
    main()