import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type="RNN", fc_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        rnn_module = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}.get(rnn_type, nn.RNN)
        self.rnn = rnn_module(hidden_size, hidden_size, batch_first=True)
        self.fc = []
        self.fc_layers = fc_layers
        self.rnn_type = rnn_type
        if fc_layers > 1:
            for _ in range(fc_layers-1):
                self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        x = torch.relu(self.fc[0](output[:, -1, :]))
        if self.fc_layers > 1:
            for i in range(self.fc_layers-1):
                x = self.fc[i+1](x)
        return x
    
    def get_rnn_type(self):
        return self.rnn_type

    def get_hidden_size(self):
        return self.hidden_size
    
    def get_fc_layers(self):
        return self.fc_layers
    
class ShakespeareDataset(Dataset):
    def __init__(self, text, char_to_ix, seq_length=20):
        self.text = text
        self.char_to_ix = char_to_ix
        self.seq_length = seq_length
        self.data = []
        self.labels = []

        for i in range(len(text) - seq_length):
            seq = text[i:i + seq_length]
            label = text[i + seq_length]
            self.data.append([char_to_ix[char] for char in seq])
            self.labels.append(char_to_ix[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def model_size(model, shared_params, shared_size, model_num):
    shared_params[model_num] = sum(p.numel() for p in model.parameters())
    shared_size[model_num] = shared_params[model_num] * 4 / (1024 * 1024)
    
def train_model(model, train_loader, val_loader, model_num, shared_training_loss, shared_validation_loss, shared_acc, shared_time):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_output = model(X_val)
                loss = criterion(val_output, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(val_output, 1)
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        if (epoch + 1) % 10 == 0:
            print(f'Model {model_num+1}, Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    shared_training_loss[model_num] = avg_train_loss
    shared_validation_loss[model_num] = avg_val_loss
    shared_acc[model_num] = val_accuracy
    shared_time[model_num] = time.time() - start_time

def predict_next_char(model, char_to_ix, ix_to_char, initial_str, max_length):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

hidden_states = []
learning_rate = 0.001
epochs = 100
batch_size = 4096
hidden_size = [128, 256, 512]
fc_layers = [1, 2, 3]


if __name__ == '__main__':

    with open('C:/Users/pengo/tinyShakeSpeare.txt', 'r') as f:
        text = f.read()

    model_Iter = ["LSTM", "GRU"]
    max_length = [20, 30, 50]

    process_num = -1
    xy_list_num = -1
    model_num = -1
    length_num = -1

    models = []
    p = []
    model_length = []
    predicted_char_results = []
    csv_data = []
    train_dataset_list, train_loader_list, val_loader_list = [], [], []

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
        length_num += 1
        train_dataset_list.append(ShakespeareDataset(text, char_to_ix, i))
        train_loader_list.append(DataLoader(train_dataset_list[length_num], batch_size=batch_size, shuffle=True))
        val_loader_list.append(DataLoader(train_dataset_list[length_num], batch_size=batch_size, shuffle=False))

    for i in max_length:
        xy_list_num += 1
        for j in model_Iter:
            for k in hidden_size:
                for l in fc_layers:
                    models.append(CharRNN(len(chars), hidden_size[k], len(chars), j, fc_layers[k]).to(device))
                    model_length.append(max_length[xy_list_num])
                    process_num += 1
                    train_model(models[process_num], train_loader_list[xy_list_num], val_loader_list[xy_list_num], process_num, training_loss_results, validation_loss_results, validation_accuracy_results, training_time_results)

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
        print(f"Model {model_num+1}, RNN Type {i.get_rnn_type()}, Sequence Length {model_length[model_num]}, Training Loss: {training_loss_results[model_num]}, Validation Loss: {validation_loss_results[model_num]}, Validation Accuracy: {validation_accuracy_results[model_num]}, Training time: {training_time_results[model_num]}, Inference time:  {inference_time_results[model_num]}, Parameters: {model_parameters_results[model_num]}, Size: {model_size_results[model_num]} MB, Predicted next character: '{predicted_char_results[model_num]}'")
        model_data = {
            "Model": model_num + 1,
            "RNN Type": i.get_rnn_type(),
            "Sequence Length": model_length[model_num],
            "Training Loss": training_loss_results[model_num],
            "Validation Loss": validation_loss_results[model_num],
            "Validation Accuracy": validation_accuracy_results[model_num],
            "Training time": training_time_results[model_num],
            "Inference Time": inference_time_results[model_num],
            "Hidden Size": i.get_hidden_size(),
            "Fully Connected Layers": i.get_fc_layers(),
            "Parameters": model_parameters_results[model_num],
            "Size (MB)": model_size_results[model_num],
            "Predicted next character": predicted_char_results[model_num]
        }
        
        csv_data.append(model_data)

    header = [
        "Model", "RNN Type", "Sequence Length", "Training Loss", "Validation Loss", "Validation Accuracy", 
        "Training time", "Inference Time", "Hidden Size", "Fully Connected Layers", "Parameters", "Size (MB)", "Predicted next characters"
    ]

    with open('model_results_spear.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_data)

