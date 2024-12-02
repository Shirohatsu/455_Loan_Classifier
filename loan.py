import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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


def train(model, X_train, Y_train, epochs, loss_function, optimizer):

    for epoch in tqdm(range(epochs)):

        outputs = model(X_train)
        loss = loss_function(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


def evaluation(model, X_test, Y_test):

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == Y_test).float().mean()
        print(f'Test Accuracy: {accuracy.item():.4f}')

    return predictions




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
        epochs = 10_000,
        loss_function = nn.BCELoss(),
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    )

    predictions = evaluation(
        model = model,
        X_test = test_x,
        Y_test = test_y
    )



if __name__ == '__main__':

    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    # main()