import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torchinfo import summary
from sklearn.model_selection import train_test_split
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type="RNN"):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        rnn_module = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}.get(rnn_type, nn.RNN)
        self.rnn = rnn_module(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output
    
def model_size(model, shared_params, shared_size, model_num):
    shared_params[model_num] = sum(p.numel() for p in model.parameters())
    shared_size[model_num] = shared_params[model_num] * 4 / (1024 * 1024)
    
def train_model(model, X_train, y_train, X_val, y_val, model_num, shared_training_loss, shared_validation_loss, shared_acc, shared_time):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            _, predicted = torch.max(val_output, 1)
            val_accuracy = (predicted == y_val).float().mean()
        
        if (epoch+1) % 10 == 0:
            print(f'Model {model_num+1}, Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')
        
    shared_training_loss[model_num] = loss.item()
    shared_validation_loss[model_num] = val_loss.item()
    shared_acc[model_num] = val_accuracy.item()
    shared_time[model_num] = time.time() - start_time

def predict_next_char(model, char_to_ix, ix_to_char, initial_str, max_length):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

hidden_size = 64
learning_rate = 0.001
epochs = 100

if __name__ == '__main__':

    with open('tinyShakeSpeare.txt', 'r') as f:
        text = f.read()

    model_Iter = ["LSTM", "GRU"]
    max_length = [20, 30, 50]

    process_num = -1
    xy_list_num = -1
    model_num = -1

    models = []
    p = []
    model_length = []
    predicted_char_results = []
    csv_data = []
    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []

    chars = sorted(list(set(text)))
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    chars = sorted(list(set(text)))

    training_loss_results = [None] * 9
    validation_loss_results = [None] * 9
    validation_accuracy_results = [None] * 9
    training_time_results = [None] * 9
    inference_time_results  = []
    model_parameters_results = [None] * 9
    model_size_results = [None] * 9
    
    for i in max_length:
        X = []
        y = []
        
        for j in range(len(text) - i):
            sequence = text[j:j + i]
            label = text[j + i]
            X.append([char_to_ix[char] for char in sequence])
            y.append(char_to_ix[label])
        
        X = np.array(X)
        y = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_list.append(torch.tensor(X_train, dtype=torch.long))
        y_train_list.append(torch.tensor(y_train, dtype=torch.long))
        X_val_list.append(torch.tensor(X_val, dtype=torch.long))
        y_val_list.append(torch.tensor(y_val, dtype=torch.long))

    for i in max_length:
        xy_list_num += 1
        for j in model_Iter:
            models.append(CharRNN(len(chars), hidden_size, len(chars), j).to(device))
            model_length.append(max_length[xy_list_num])
            process_num += 1
            train_model(models[process_num], X_train_list[xy_list_num], y_train_list[xy_list_num], X_val_list[xy_list_num], y_val_list[xy_list_num], process_num, training_loss_results, validation_loss_results, validation_accuracy_results, training_time_results)

    test_str = (
        """
        WARWICK:
        So much his friend, ay, his unfeigned friend,
        That, if King Lewis vouchsafe to furnish us
        With some few bands of chosen soldiers,
        I'll undertake to land them on our coast
        And force the tyrant from his seat by war.
        'Tis not his new-made bride shall succor him:
        And as for Clarence, as my letters tell me,
        He's very likely now to fall from him,
        For matching more for wanton lust than honour,
        Or than for strength and safety of our country.
        """
    )

    for i in models:
        model_num += 1
        start_time = time.time()
        generated_sequence = test_str
        max_sequence_length = 100

        for _ in range(max_sequence_length):
            predicted_char = predict_next_char(i, char_to_ix, ix_to_char, generated_sequence, model_length[model_num])
            generated_sequence += predicted_char
        
        inference_time = time.time() - start_time
        inference_time_results.append(inference_time)
        predicted_char_results.append(generated_sequence)

        model_size(i, model_parameters_results, model_size_results, model_num)
        print(f"Model {model_num+1}, Sequence Length {model_length[model_num]}, Training Loss: {training_loss_results[model_num]}, Validation Loss: {validation_loss_results[model_num]}, Validation Accuracy: {validation_accuracy_results[model_num]}, Training time: {training_time_results[model_num]}, Inference time:  {inference_time_results[model_num]}, Parameters: {model_parameters_results[model_num]}, Size: {model_size_results[model_num]} MB, Predicted next character: '{predicted_char_results[model_num]}'")
        model_data = {
            "Model": model_num + 1,
            "Sequence Length": model_length[model_num],
            "Training Loss": training_loss_results[model_num],
            "Validation Loss": validation_loss_results[model_num],
            "Validation Accuracy": validation_accuracy_results[model_num],
            "Training time": training_time_results[model_num],
            "Inference Time": inference_time_results[model_num],
            "Parameters": model_parameters_results[model_num],
            "Size (MB)": model_size_results[model_num],
            "Predicted next character": predicted_char_results[model_num]
        }
        
        csv_data.append(model_data)

    header = [
        "Model", "Sequence Length", "Training Loss", "Validation Loss", "Validation Accuracy", 
        "Training time", "Inference Time", "Parameters", "Size (MB)", "Predicted next characters"
    ]

    with open('model_results_spear.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_data)

