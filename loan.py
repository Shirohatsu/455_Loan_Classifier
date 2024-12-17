import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

torch.random.manual_seed(0)


def pre_process(gender_map, eduction_map, home_ownership_map, loan_intent_map, previous_loan_defaults_on_file):

    """


    :param gender_map: Binary Encoding
    :param eduction_map: Ordeal Encoding
    :param home_ownership_map: Hot Encoding
    :param loan_intent_map: Hot Encoding
    :param previous_loan_defaults_on_file: Binary Encoding
    :return:  Create a xlxs files
    """
    data = pd.read_csv('loan_data.csv')

    data['person_gender'] = data['person_gender'].map(gender_map)
    data['person_education'] = data['person_education'].map(eduction_map)
    data['person_home_ownership'] = data['person_home_ownership'].map(home_ownership_map)
    data['previous_loan_defaults_on_file'] =  data['previous_loan_defaults_on_file'].map(previous_loan_defaults_on_file)

    df_encoded = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], dtype=int)
    df_encoded.to_excel("converted.xlsx", index=False)

def read_pre_process_data(scaler):

    data = pd.read_csv('converted.csv')
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    X = scaler.fit_transform(X)

    return X, Y



def graph_data():
    """Read Loan data seet and find the most important features."""

    data = pd.read_csv('loan_data.csv')
    usable_data = gather_data(data)


def gather_data(data):

    """
    :param data: The actual Data
    :return:  Return the number of female and male along with loan approval and same with education level
    """
    gender_graph = {'female': {0: 0, 1: 0}, 'male': {0: 0, 1: 0}}
    eduction_graph = {"Bachelor": {0: 0, 1: 0}, "Associate": {0: 0, 1: 0},
                      "High School": {0: 0, 1: 0}, "Master": {0: 0, 1: 0}, "Doctorate": {0: 0, 1: 0}}
    approved  = {0: 0, 1: 0}
    for i in tqdm(range(len(data['person_gender']))):
        eduction_level = data['person_education'][i]
        approved_status = int(data['loan_status'][i])
        gender_classification = data['person_gender'][i]
        gender_graph[gender_classification][approved_status] += 1
        eduction_graph[eduction_level][approved_status] += 1
        approved[approved_status] += 1
    return [gender_graph, eduction_graph]


def model_preparation(X, Y):
    """
    :param X: All the training data excluding it classification
    :param Y: ALl the classification of the training data
    :return: The tensor version of the data to use with py torch
    """
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=0)
    train_X_tensor = torch.FloatTensor(train_X)
    train_Y_tensor = torch.FloatTensor(train_Y).unsqueeze(1)
    test_X_tensor = torch.FloatTensor(test_X)
    test_Y_tensor = torch.FloatTensor(test_Y).unsqueeze(1)
    return train_X_tensor, train_Y_tensor, test_X_tensor, test_Y_tensor

# From pytorch doc https://pytorch.org/tutorials/beginner/basics/quickstart_tutoria l.html

def train(model, X_train, Y_train, X_Test, Y_Test, epochs, loss_function, optimizer):

    """

    :param model: The ANN model it self
    :param X_train:  All the training data excluding it classification
    :param Y_train: ALl the classification of the training data
    :param X_Test: ALl the testing data excluding it classification
    :param Y_Test:  All the classification of the testing data
    :param epochs:  The number epochs
    :param loss_function:  THe loss function to use
    :param optimizer:  The step / optimizer to use
    :return:
    """
    number_of_epoch = []
    number_of_correct_guess_train = []
    number_of_correct_guess_test = []
    train_loss = []
    test_loss = []

    # Create a progress bar
    for epoch in tqdm(range(epochs)):

        outputs = model(X_train)
        loss = loss_function(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gather data about loss function and it rate
        if epoch % 10 == 0:

            model.eval()
            correct_guess_train = 0
            correct_guess_test = 0

            with torch.no_grad():
                for index, value in enumerate(outputs):
                    prediction = 1 if value > 0.5 else 0
                    if prediction == Y_train[index]:
                        correct_guess_train += 1


                for index, value in enumerate(X_Test):
                    prediction = 1 if model.forward(value) > 0.5 else 0
                    if prediction == Y_Test[index]:
                        correct_guess_test += 1

            test_output = model(X_Test)
            test_loss_value = loss_function(test_output, Y_Test)

            train_loss.append(loss.item())
            test_loss.append(test_loss_value.item())
            number_of_correct_guess_train.append(correct_guess_train / len(X_train))
            number_of_correct_guess_test.append(correct_guess_test / len(X_Test))
            number_of_epoch.append(epoch)
            model.train()



    # Graph the loss functions

    plt.plot(number_of_epoch, number_of_correct_guess_train, label="Training Accuracy")
    plt.plot(number_of_epoch, number_of_correct_guess_test, label = "Testing Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy during Training vs Test")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.show()


    plt.plot(number_of_epoch, train_loss, label="Train")
    plt.plot(number_of_epoch, test_loss, label = "Test")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()


    return model

# From the same pytorch Doc
def evaluation(model, X_test, Y_test):

    model.eval()
    correct = 0
    confusion_matrix = {'actual_approved': {'predicted_approved': 0, 'predicted_rejected': 0}, 'actual_rejected':  {'predicted_approved': 0, 'predicted_rejected': 0}}

    # Calculate the confusion matrix
    with torch.no_grad():
        for index, value in enumerate(X_test):
            prediction = 1 if model.forward(value) > 0.5 else 0
            if prediction == Y_test[index]:
                correct += 1
            if prediction == 1 and Y_test[index] == 1:
                confusion_matrix['actual_approved']['predicted_approved'] += 1

            if prediction == 0 and Y_test[index] == 1:
                confusion_matrix['actual_approved']['predicted_rejected'] += 1

            if prediction == 1 and Y_test[index] == 0:
                confusion_matrix['actual_rejected']['predicted_approved'] += 1

            if prediction == 0 and Y_test[index] == 0:
                confusion_matrix['actual_rejected']['predicted_rejected'] += 1



    # Created from using claude with the prompt
    # {'actual_approved': {'predicted_approved': 1465, 'predicted_rejected': 524},
    # 'actual_rejected': {'predicted_approved': 463, 'predicted_rejected':
    # how can I make a confusion matrix out of this


    # Then I modified the np.array to make it more readable and added f1 scores

    cm = np.array([
        [confusion_matrix['actual_approved']['predicted_approved'], confusion_matrix['actual_approved']['predicted_rejected']],
        [confusion_matrix['actual_rejected']['predicted_approved'], confusion_matrix['actual_rejected']['predicted_rejected']],
    ])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar=False,  #
                xticklabels=['Predicted\nApproved', 'Predicted\nRejected'],
                yticklabels=['Actual\nApproved', 'Actual\nRejected'])

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.show()

    total_samples = np.sum(cm)
    accuracy = np.trace(cm) / total_samples
    precision_approved = cm[0, 0] / (cm[0, 0] + cm[1, 0]) # TP / TP + FP
    precision_rejected = cm[1, 1] / (cm[1, 0] + cm[1, 1]) # TF / TF + FN
    recall_approved = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall_rejected = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    f1_score_approved =  2 * (precision_approved * recall_approved) / (precision_approved + recall_approved)
    f1_score_rejected = 2 * (precision_rejected * recall_rejected) / (precision_rejected + recall_rejected)

    print(f"Total Samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Approved): {precision_approved:.4f}")
    print(f"Precision (Rejected): {precision_rejected:.4f}")
    print(f"Recall (Approved): {recall_approved:.4f}")
    print(f"Recall (Rejected): {recall_rejected:.4f}")
    print(f"F1 Score (Approved): {f1_score_approved:.4f}")
    print(f"F1 Score (Rejected): {f1_score_rejected:.4f}")

    print(f"Testing Accuracy {correct / len(X_test)}")

    return correct / len(X_test)



# From https://jillanisofttech.medium.com/building-an-ann-with-pytorch-a-deep-dive-into-neural-network-training-a7fdaa047d81
"""
Citation: 
Tech, J. S. (2024b, January 8). Building an ann with pytorch: A deep dive into neural network training 🚀. Medium. 
https://jillanisofttech.medium.com/building-an-ann-with-pytorch-a-deep-dive-into-neural-network-training-a7fdaa047d81 
"""
class LoanApproval(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 28)
        self.fc2 = nn.Linear(28, 14)
        self.output = nn.Linear(14, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

def main():

    # The old of converting data to usable stuff
    # Only ran once, and it generates a file named converted.xlsx
    # Open that files and move loan status to the end  that it
    # gender_map = {"female": 0, "male": 1}
    # education_map = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
    # home_ownership_map = {"OTHER": 1, "RENT": 2, "MORTGAGE": 3, "OWN": 4}
    # previous_loan_defaults_on_file = {'Yes': 1, 'No': 0}
    # loan_intent_map = {}


    # pre_process(
    #     gender_map = gender_map,
    #     eduction_map = education_map,
    #     home_ownership_map =home_ownership_map,
    #     loan_intent_map = loan_intent_map,
    #     previous_loan_defaults_on_file = previous_loan_defaults_on_file,
    # )
    #
    X , Y = read_pre_process_data(StandardScaler())
    train_x, train_y, test_x, test_y = model_preparation(X, Y)

    model = LoanApproval(train_x.shape[1])
    # Train the model
    train(
        model = model, # The model
        X_train = train_x, # THe Train Data
        Y_train = train_y, # The classification
        X_Test = test_x, # The test data
        Y_Test = test_y, # The classification
        epochs = 1_000, # THe number if epochs
        loss_function = nn.BCELoss(), # The loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001) # The optimizer
    )

    model.eval()


    predictions = evaluation(

        model = model,
        X_test = test_x,
        Y_test = test_y
    )



if __name__ == '__main__':
    # graph_data()


    main()
