import pandas as pd
import numpy as np
import time

def main():
    file = str(input("Welcome to my Feature Selection Algorithm!\nPlease type in the name of the file to test the algorithm (using format: name_of_file.txt): \n"))
    selection = input("\nNow, please type in the number of the feature selection algorithm you want to run:\n\n1) Forwards Selection\n2) Backwards Selection\n")
    df = pd.read_csv(file, delimiter=r"\s+", header=None, dtype=float)

    # uncomment to run excel file (for extra credit)
    # df = pd.read_excel(file, engine="openpyxl")
    print("This dataset has " + str(df.shape[1]-1) + " columns, not including the class attribute. It has " + str(df.shape[0]) + " instances.")
    # forward selection
    if selection == '1':
        # base accuracy, no features
        base_accuracy = get_accuracy(df,[])
        print(f"\nRunning nearest neighbor with no features, using 'leave-one-out' evaluation, I get an accuracy of {(round(base_accuracy*100, 2))}%")
        print("Beginning search.")
        forward_selection(df, base_accuracy)
    # backward selection
    else:
        # base accuracy, all features
        base_accuracy = get_accuracy(df,list(range(1, df.shape[1])))
        print(f"\nRunning nearest neighbor with all {(df.shape[1]-1)} features, using 'leave-one-out' evaluation, I get an accuracy of {(round(base_accuracy*100, 2))}%") 
        print("Beginning search.")
        backward_selection(df, base_accuracy)

# both forward and backward selection built based on lecture slides/Dr.Keogh's lecture videos

def forward_selection(df, min_accuracy):
    # initializing variables
    begin = time.perf_counter() # start time taken 
    highest_accuracy_features = [] # stores final chosen features 
    current_features = [] # stores max feature per iteration -- might not be a part of final set
    max_accuracy = min_accuracy

    # for graphs, not printed anywhere
    all_max_accuracies = []
    all_max_features = []

    # loops through all features except target
    for i in range(1, df.shape[1]):
        print(f"\nOn level {i} of the search tree")
        local_max_feature = None
        local_max_accuracy = 0

        # loop through all features except target
        for j in range(1, df.shape[1]):
            if(j in current_features or j in highest_accuracy_features):
                continue
            # calculate accuracy if adding current feature increases accuracy
            accuracy = get_accuracy(df, list(current_features) + [j])
            print(f"Using feature(s) {current_features+ [j]}, accuracy is {round(accuracy*100, 2)}%")
            if accuracy > local_max_accuracy:
                local_max_accuracy = accuracy
                local_max_feature = j     

        # append the local maximum feature, even if it isn't the best accuracy -- ensures that there's no infinite loop by continuing the greedy algorithm
        if local_max_feature != None:
            current_features.append(local_max_feature)
            get_max_accuracy = round(local_max_accuracy*100,2)
            all_max_accuracies.append(get_max_accuracy)
            all_max_features.append(local_max_feature)
        
        if local_max_accuracy < max_accuracy:
            print("(Warning, the overall accuracy has decreased. Continuing search in case of local maxima!)")
            continue
        # if the local max is more than our actual max, then we should make that the set of features
        else:
            max_accuracy = local_max_accuracy
            highest_accuracy_features = current_features.copy()
            print(f"Feature set {highest_accuracy_features} was best with accuracy of {round(max_accuracy*100, 2)}%")
    if not highest_accuracy_features:
        print("Forward selection could not find a subset of features that had a higher accuarcy than the base model.")
        return
    print(f"\nSuccess! The best feature subset is {highest_accuracy_features}, with an accuracy of {round(max_accuracy*100, 2)}%! The time taken is {round(time.perf_counter() - begin, 3)} seconds.")


# nearly the same as forward selection, added comments where the code differs
def backward_selection(df, min_accuracy):
    begin = time.perf_counter()
    highest_accuracy_features = []
    # current_features = df.columns.tolist()[1:] # take all set of features

    num_columns = len(df.columns) - 1  # Get total columns minus 1
    current_features = list(range(1, num_columns + 1))     
    max_accuracy = min_accuracy
    all_max_accuracies = []
    all_max_features = []
    for i in range(1, df.shape[1]):
        print(f"\nOn level {i} of the search tree")
        local_max_feature = None
        local_max_accuracy = 0
        for j in range(1, df.shape[1]):
            # check to see if current feature has not be REMOVED, rather than if it has been added
            if(j not in current_features):
                continue
            accuracy = get_accuracy(df, [col for col in current_features if col != j])
            selected_features = current_features.copy()
            selected_features.remove(j)
            print(f"Using feature(s) {selected_features}, accuracy is {round(accuracy*100, 2)}%")
            if accuracy > local_max_accuracy:
                local_max_accuracy = accuracy
                local_max_feature = j
        # remove the local maximum feature, even if it isn't the best accuracy -- ensures that there's no infinite loop by continuing the greedy algorithm
        current_features.remove(local_max_feature)
        get_max_accuracy = round(local_max_accuracy*100,2)
        all_max_accuracies.append(get_max_accuracy)
        all_max_features.append(local_max_feature)
        if local_max_accuracy < max_accuracy:
            print("(Warning, the overall accuracy has decreased. Continuing search in case of local maxima!)")
            continue
        # if the local max is more than our actual max, then we should make that the set of features
        else:
            max_accuracy = local_max_accuracy
            highest_accuracy_features = current_features.copy()
            print(f"Feature set {highest_accuracy_features} was best with accuracy of {round(max_accuracy*100, 2)}%")
    if not highest_accuracy_features:
        print("Backward selection could not find a subset of features that had a higher accuarcy than the base model.")
        return
    print(f"\nSuccess! The best feature subset is {highest_accuracy_features}, with an accuracy of {round(max_accuracy*100, 2)}%! The time taken is {round(time.perf_counter() - begin, 3)} seconds.")

def get_accuracy(df, current_features = None):
    # using numpy since it is more efficient for performing matrix calculations
    # resources: https://stackoverflow.com/questions/52670767/summing-array-values-over-a-specific-range-with-numpy
        # https://numpy.org/doc/2.1/reference/generated/numpy.array.html
        # https://www.geeksforgeeks.org/python-numpy/
    data = df.to_numpy()
    labels = data[:, 0]
    current_features = [int(f) for f in current_features]
    # getting the features to check for accuracy
    features = data[:, current_features]

    correctly_classified = 0
    num_samples = features.shape[0]

    # looping through all rows
    for i in range(num_samples):
        obj_to_classify = features[i] # row i
        distances = np.sqrt(np.sum((features - obj_to_classify) ** 2, axis=1)) # get distance from all other rows
        distances[i] = np.inf # remove case of self
        nn_index = np.argmin(distances)  # smallest distance = label that we want
        nn_label = labels[nn_index]
        if nn_label == labels[i]:
            correctly_classified += 1
    accuracy = correctly_classified / num_samples
    return accuracy

if __name__ == "__main__":
    main()