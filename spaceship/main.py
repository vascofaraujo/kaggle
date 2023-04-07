import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb


def get_data(files, is_train):

    # Drop useless categories if training
    if is_train:
        files = files.drop(["PassengerId","Name"],axis=1)
    else:
        files = files.drop(["Name"],axis=1)

    # Join luxuries in single column, split cabin into deck, num and side
    files["TotalSpent"] = files["RoomService"] + files["FoodCourt"] + files["ShoppingMall"] + files["Spa"] + files["VRDeck"]
    files[['Deck','Num','Side']] = files["Cabin"].str.split('/',expand=True)
    files = files.drop(["RoomService","FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Cabin"],axis=1)

    # You can't spend money on cryosleep
    files.loc[files["CryoSleep"].isin([True]), "TotalSpent"] = 0

    # Fill missing values
    files["HomePlanet"].fillna(files["HomePlanet"].mode()[0], inplace=True)
    files["CryoSleep"].fillna(files["CryoSleep"].mode()[0], inplace=True)
    files["Destination"].fillna(files["Destination"].mode()[0], inplace=True)
    files["HomePlanet"].fillna(files["HomePlanet"].mode()[0], inplace=True)
    files["Age"].fillna(files["Age"].mean(), inplace=True)
    files["VIP"].fillna(files["VIP"].mode()[0], inplace=True)
    files["TotalSpent"].fillna(files["TotalSpent"].mode()[0], inplace=True)

    # Drop Nulls
    if is_train:
        files.dropna(inplace=True)

    # Encode strings into ints
    categoricals = files.select_dtypes(exclude=[np.number])
    print(categoricals)
    for column in categoricals.columns:
        if column == "PassengerId":
            pass
        unique_string = {}
        # new_column = df
        unique_int = 0
        for x in files[column]:
            if x not in unique_string:
                unique_string[x] = unique_int
                unique_int += 1

        files[column] = files[column].replace(unique_string)
    return files



def train(train_files):
    X, y = train_files.values[:, :-1], train_files.values[:, -1]


    model = xgb.XGBClassifier(n_estimators=10000)
    model.fit(X, y)

    dtrain = xgb.DMatrix(X, label=y)
    print(dtrain)
    param = {"objective": "reg:squarederror"}
    num_round = 1000
    bst = xgb.train(param, dtrain, num_round)
    print(bst)
    # best_model =
    return bst
    # return best_model


def predict(model, test_files):
    X, y = train_files.values[:, :-1], train_files.values[:, -1]


    dtest = xgb.DMatrix(X, label=y)

    y_hat = model.predict(dtest)
    test_transported = pd.Series(y_hat, name="Transported").astype("bool")
    results = pd.concat([test_files["PassengerId"], test_transported], axis=1)
    results.to_csv("spaceship2.csv", index=False)


if __name__ == "__main__":
    train_files = get_data(pd.read_csv("train.csv"), is_train=True)
    test_files = get_data(pd.read_csv("test.csv"), is_train=False)

    model = train(train_files)
    predictions = predict(model, test_files)


