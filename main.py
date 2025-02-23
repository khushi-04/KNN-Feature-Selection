import pandas as pd
import numpy as np

def main():
    file = str(input("Welcome to my Feature Selection Algorithm!\nPlease type in the name of the file to test the algorithm (using format: name_of_file.txt): \n"))
    selection = input("\nNow, please type in the number of the feature selection algorithm you want to run:\n\n1) Forwards Selection\n2) Backwards Selection\n")
    # file = "KNN-Feature-Selection\\" + file
    df = pd.read_csv(file, delimiter=r"\s+", header=None, dtype=float)

    print("This dataset has " + str(df.shape[1]-1) + " columns, not including the class attribute. It has " + str(df.shape[0]) + " instances.")

    forward_selection(df)
    # df[1] = 0


def forward_selection(df):
    current_features = []
    for i in range(1, df.shape[1]):
        print("On the " + str(i) + "th level of the search tree")
        feature = None
        max_accuracy = 0
        for j in range(1, df.shape[1]):
            if(j not in current_features):
                print("Considering adding " + str(j) + " feature")
                accuracy = get_accuracy(df, current_features, j)
                # print(accuracy)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    feature = j
        current_features.append(feature)
        print("On level " + str(i) + ", I added feature " + str(feature))

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


def get_accuracy(df, current_features, feature):
    df_copy = df.copy()
    correctly_classified = 0
    for i in range(df_copy.shape[1]):
        if i not in current_features and i != feature:
            df_copy[i] = 0
    for i in range(1, df_copy.shape[0]):
        obj_to_classify = df_copy.iloc[i,1:].values
        label_obj_to_classify = df_copy.iloc[i,0]
        nn_dist = np.inf
        nn_location = np.inf
        for j in range(1, df_copy.shape[0]):
            if j != i:
                distance = np.sqrt(np.sum((obj_to_classify - df_copy.iloc[j, 1:].values) ** 2))
                if distance < nn_dist:
                    nn_dist = distance
                    nn_location = j
                    nn_label = df_copy.iloc[nn_location,0]
        if nn_label == label_obj_to_classify:
            
            correctly_classified = correctly_classified + 1
    return (correctly_classified / df_copy.shape[0])


    # return 0


    

if __name__ == "__main__":
    main()