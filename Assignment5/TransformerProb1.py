import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import multiprocessing
from torchinfo import summary
from sklearn.model_selection import train_test_split
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output
    
def model_size(model, shared_params, shared_size, model_num):
    shared_params[model_num] = sum(p.numel() for p in model.parameters())
    shared_size[model_num] = shared_params[model_num] * 4 / (1024 * 1024)
    
def train_model(model, X_train, y_train, X_val, y_val, model_num, shared_loss, shared_acc, shared_time):
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
        
    end_time = time.time()
    shared_loss[model_num] = loss.item()
    shared_acc[model_num] = val_accuracy.item()
    shared_time[model_num] = end_time - start_time

def predict_next_char(model, char_to_ix, ix_to_char, initial_str, max_length):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

hidden_size = 256
learning_rate = 0.001
epochs = 100

if __name__ == '__main__':

    text = (
        """
        Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

        At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

        One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

        Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

        Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

        In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."
        """
    )

    model_Iter = ["Transformer"]
    max_length = [10, 20, 30]
    nhead = 2
    numLayers = 2

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

    manager = multiprocessing.Manager()
    training_loss_results = manager.list([None] * 9)
    validation_accuracy_results = manager.list([None] * 9)
    training_time_results = manager.list([None] * 9)
    model_parameters_results = manager.list([None] * 9)
    model_size_results = manager.list([None] * 9)
    
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
        models.append(CharTransformer(len(chars), hidden_size, len(chars), numLayers, nhead).to(device))
        model_length.append(max_length[xy_list_num])
        process_num += 1
        p.append(multiprocessing.Process(target=train_model, args=(models[process_num], X_train_list[xy_list_num], y_train_list[xy_list_num], X_val_list[xy_list_num], y_val_list[xy_list_num], process_num, training_loss_results, validation_accuracy_results, training_time_results)))
        p[process_num].start()

    for i in p:
        i.join()

    test_str = (
        """
        One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

        Training a model for next character predic
        """
    )

    for i in models:
        model_num += 1
        predicted_char = predict_next_char(i, char_to_ix, ix_to_char, test_str, model_length[model_num])
        predicted_char_results.append(predicted_char)
        model_size(i, model_parameters_results, model_size_results, model_num)
        print(f"Model {model_num+1}, Sequence Length {model_length[model_num]}, Loss: {training_loss_results[model_num]}, Validation Accuracy: {validation_accuracy_results[model_num]}, Training time: {training_time_results[model_num]}, Parameters: {model_parameters_results[model_num]}, Size: {model_size_results[model_num]} MB, Predicted next character: '{predicted_char_results[model_num]}'")
        model_data = {
            "Model": model_num + 1,
            "Sequence Length": model_length[model_num],
            "Loss": training_loss_results[model_num],
            "Validation Accuracy": validation_accuracy_results[model_num],
            "Training time": training_time_results[model_num],
            "Parameters": model_parameters_results[model_num],
            "Size (MB)": model_size_results[model_num],
            "Predicted next character": predicted_char_results[model_num]
        }
        
        csv_data.append(model_data)

    header = [
        "Model", "Sequence Length", "Loss", "Validation Accuracy", 
        "Training time", "Parameters", "Size (MB)", "Predicted next character"
    ]

    with open('model_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_data)

