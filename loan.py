import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def pre_process(gender_map, eduction_map, home_ownership_map, loan_intent_map, previous_loan_defaults_on_file, scaler):

    data = pd.read_csv('loan_data.csv')

    data['person_gender'] = data['person_gender'].map(gender_map)
    data['person_education'] = data['person_education'].map(eduction_map)
    data['person_home_ownership'] = data['person_home_ownership'].map(home_ownership_map)

    if len(loan_intent_map) != 0:
        data['loan_intent'] = data['loan_intent'].map(loan_intent_map)
    else:
        data['loan_intent'] = pd.Categorical(data['loan_intent']).codes

    data['previous_loan_defaults_on_file'] =  data['previous_loan_defaults_on_file'].map(previous_loan_defaults_on_file)

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    X = scaler.fit_transform(X)
    return X, Y


def model_preparation(X, Y):

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=0)

    train_X_tensor = torch.FloatTensor(train_X)
    train_Y_tensor = torch.FloatTensor(train_Y).unsqueeze(1)
    test_X_tensor = torch.FloatTensor(test_X)
    test_Y_tensor = torch.FloatTensor(test_Y).unsqueeze(1)
    return train_X_tensor, train_Y_tensor, test_X_tensor, test_Y_tensor

# From pytorch doc https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

def train(model, X_train, Y_train, epochs, loss_function, optimizer):
    number_of_epoch = []
    number_of_correct_guess = []
    loss_rate = []
    for epoch in tqdm(range(epochs)):

        outputs = model(X_train)
        loss = loss_function(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            correct_guess = 0
            for index, value in enumerate(outputs):
                prediction = 1 if value > 0.5 else 0
                if prediction == Y_train[index]:
                    correct_guess += 1
            number_of_correct_guess.append(correct_guess / len(Y_train))
            number_of_epoch.append(epoch)

    print(f"Final Accuracy {number_of_correct_guess[-1]}")
    plt.plot(number_of_epoch, number_of_correct_guess, loss_rate)
    plt.xlabel("# of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs # of Epochs")
    plt.show()

    return model

# From the same pytorch Doc
def evaluation(model, X_test, Y_test):

    model.eval()
    correct = 0

    with torch.no_grad():

        for index, value in enumerate(X_test):
            prediction = 1 if model.forward(value) > 0.5 else 0
            if prediction == Y_test[index]:
                correct += 1

    print(f"Accuracy {correct / len(X_test)}")

    return correct / len(X_test)




class LoanApproval(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def main():
    gender_map = {"female": 0, "male": 1}
    education_map = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "Doctorate": 5}
    home_ownership_map = {"OTHER": 1, "RENT": 2, "MORTGAGE": 3, "OWN": 4}
    previous_loan_defaults_on_file = {'Yes': 1, 'No': 0}
    loan_intent_map = {}

    X, Y = pre_process(
        gender_map = gender_map,
        eduction_map = education_map,
        home_ownership_map =home_ownership_map,
        loan_intent_map = loan_intent_map,
        previous_loan_defaults_on_file = previous_loan_defaults_on_file,
        scaler = MinMaxScaler()
    )

    train_x, train_y, test_x, test_y = model_preparation(X, Y)

    model = LoanApproval(train_x.shape[1])
    train(
        model = model,
        X_train = train_x,
        Y_train = train_y,
        epochs = 10_00,
        loss_function = nn.BCELoss(),
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    )

    predictions = evaluation(

        model = model,
        X_test = test_x,
        Y_test = test_y
    )



if __name__ == '__main__':

    main()