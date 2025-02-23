import pandas as pd
def main():
    file = str(input("Welcome to my Feature Selection Algorithm!\nPlease type in the name of the file to test the algorithm (using format: name_of_file.txt): \n"))
    selection = input("\nNow, please type in the number of the feature selection algorithm you want to run:\n\n1) Forwards Selection\n2) Backwards Selection\n")
    file = "KNN-Feature-Selection\\" + file
    df = pd.read_csv(file, delimiter=r"\s+", header=None, dtype=float)

    print("This dataset has " + str(df.shape[1]-1) + " columns, not including the class attribute. It has " + str(df.shape[0]) + " instances.")

    forward_selection(df)
    # df[1] = 0


def forward_selection(df):
    current_features = []
    for i in range(df.shape[1]-1):
        print("On the " + str(i) + "th level of the search tree")
        feature = None
        max_accuracy = 0
        for j in range(df.shape[1]-1):
            if(j not in current_features):
                print("Considering adding " + str(j) + " feature")
                accuracy = 1
                # accuracy = accuracy(df, current_features, j)
                get_accuracy(df, current_features, j)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    feature = j
        current_features.append(feature)
        print("On level " + str(i) + ", I added feature " + str(feature))

def backward_selection(df):
    current_features = []
    for i in range(df.shape[1]-1):
        print("On the " + str(i) + "th level of the search tree")
        feature = None
        max_accuracy = 0
        for j in range(df.shape[1]-1):
            if(j not in current_features):
                print("Considering adding " + str(j) + " feature")
                # accuracy = accuracy(df, current_features, j)
                get_accuracy(df, current_features, j)
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
    print(df_copy.head())

    # return 0


    

if __name__ == "__main__":
    main()