import pandas as pd
import numpy as np

def main():
    file = str(input("Welcome to my Feature Selection Algorithm!\nPlease type in the name of the file to test the algorithm (using format: name_of_file.txt): \n"))
    selection = input("\nNow, please type in the number of the feature selection algorithm you want to run:\n\n1) Forwards Selection\n2) Backwards Selection\n")
    df = pd.read_csv(file, delimiter=r"\s+", header=None, dtype=float)
    print("This dataset has " + str(df.shape[1]-1) + " columns, not including the class attribute. It has " + str(df.shape[0]) + " instances.")
    base_accuracy = get_accuracy(df)
    print("\nRunning nearest neighbor with all " + str(df.shape[1]-1) + " features, using 'leave-one-out' evaluation, I get an accuracy of " + str(base_accuracy)) 
    print("Starting search...")
    if selection == '1':
        forward_selection(df, base_accuracy)
    else:
        backward_selection(df, base_accuracy)

def forward_selection(df, min_accuracy):
    highest_accuracy_features = []
    current_features = []
    max_accuracy = min_accuracy
    for i in range(1, df.shape[1]):
        print("\nOn level " + str(i) + " of the search tree")
        local_max_feature = None
        local_max_accuracy = 0
        for j in range(1, df.shape[1]):
            if(j in current_features or j in highest_accuracy_features):
                continue
            accuracy = get_accuracy(df, current_features, j, 1)
            print(f"Using feature(s) {current_features+ [j]}, accuracy is {round(accuracy*100, 2)}%")
            if accuracy > local_max_accuracy:
                local_max_accuracy = accuracy
                local_max_feature = j
        # append the local maximum feature, even if it isn't the best accuracy -- ensures that there's no infinite loop by continuing the greedy algorithm
        current_features.append(local_max_feature)
        # if the local max is more than our actual max, then we should make that the set of features
        if local_max_accuracy < max_accuracy:
            print("(Warning, the overall accuracy has decreased. Continuing search in case of local maxima!)")
            continue
        else:
            max_accuracy = local_max_accuracy
            highest_accuracy_features = current_features.copy()
            print(f"Feature set {highest_accuracy_features} was best with accuracy of {round(max_accuracy*100, 2)}%")
    if not highest_accuracy_features:
        print("Forward selection could not find a subset of features that had a higher accuarcy than the base model.")
        return
    print(f"Success! The best feature subset is {highest_accuracy_features}, with an accuracy of {round(max_accuracy*100, 2)}%!")
    print(highest_accuracy_features)

def backward_selection(df, min_accuracy):
    highest_accuracy_features = []
    current_features = df.columns.tolist()[1:]
    max_accuracy = min_accuracy
    for i in range(1, df.shape[1]):
        print("\nOn level " + str(i) + " of the search tree")
        local_max_feature = None
        local_max_accuracy = 0
        for j in range(1, df.shape[1]):
            if(j not in current_features):
                continue
            accuracy = get_accuracy(df, current_features, j, 2)
            selected_features = current_features.copy()
            selected_features.remove(j)
            print(f"Using feature(s) {selected_features}, accuracy is {round(accuracy*100, 2)}%")
            if accuracy > local_max_accuracy:
                local_max_accuracy = accuracy
                local_max_feature = j
        # remove the local maximum feature, even if it isn't the best accuracy -- ensures that there's no infinite loop by continuing the greedy algorithm
        current_features.remove(local_max_feature)
        # if the local max is more than our actual max, then we should make that the set of features
        if local_max_accuracy < max_accuracy:
            print("(Warning, the overall accuracy has decreased. Continuing search in case of local maxima!)")
            continue
        else:
            max_accuracy = local_max_accuracy
            highest_accuracy_features = current_features.copy()
            print(f"Feature set {highest_accuracy_features} was best with accuracy of {round(max_accuracy*100, 2)}%")
    if not highest_accuracy_features:
        print("Backward selection could not find a subset of features that had a higher accuarcy than the base model.")
        return
    print(f"Success! The best feature subset is {highest_accuracy_features}, with an accuracy of {round(max_accuracy*100, 2)}%!")
    print(highest_accuracy_features)


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
        selected_features = list(current_features) + [feature]
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