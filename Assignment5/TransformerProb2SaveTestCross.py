import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import csv
import torch.multiprocessing as mp
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=False
mp.set_start_method('spawn', force=True) 
checkpoint_dir = "C:/Users/Jacob/Prob3Checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

SOS_token = 0  # Start of Sequence token
EOS_token = 1  # End of Sequence token

class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.fc_layers = num_layers
        self.nhead = nhead

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        memory = self.transformer_encoder(src_emb)
        output = self.decoder(tgt_emb, memory)
        return self.fc(output[:, -1, :])

    def get_hidden_size(self):
        return self.hidden_size
    
    def get_fc_layers(self):
        return self.fc_layers
    
    def get_nhead(self):
        return self.nhead
    
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

def save_checkpoint(model, optimizer, epoch, model_num, elapsed_time, avg_train_loss, avg_val_loss, val_accuracy):
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_num}_epoch_{epoch}.pth")
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"model_{model_num}_")]
    print(existing_checkpoints)
    for checkpoint in existing_checkpoints:
        os.remove(os.path.join(checkpoint_dir, checkpoint))
        print(f"Removed previous checkpoint: {checkpoint}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'elapsed_time': elapsed_time,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, model_num):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"model_{model_num}_")]
    if not checkpoint_files:
        return 0, 0, 0, 0, 0

    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resuming training from {checkpoint_path}")

    return checkpoint['epoch'], checkpoint['elapsed_time'], checkpoint['avg_train_loss'], checkpoint['avg_val_loss'], checkpoint['val_accuracy']
    
def train_model(model, train_loader, val_loader, model_num, shared_training_loss, shared_validation_loss, shared_acc, shared_time):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    model.to(device)

    start_epoch, elapsed_time, avg_train_loss, avg_val_loss, val_accuracy = load_checkpoint(model, optimizer, model_num)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        start_epoch_time = time.time()
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            tgt = torch.cat([torch.zeros((y_batch.size(0), 1), dtype=torch.long, device=device), y_batch.unsqueeze(1)], dim=1)
            with autocast(enabled=True, dtype=torch.float16):
                output = model(X_batch, tgt)
                loss = criterion(output, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                tgt = torch.cat([torch.zeros((y_val.size(0), 1), dtype=torch.long, device=device), y_val.unsqueeze(1)], dim=1)
                with autocast(enabled=True, dtype=torch.float16):
                    val_output = model(X_val, tgt)
                    loss = criterion(val_output, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(val_output, 1)
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        epoch_time = time.time() - start_epoch_time
        elapsed_time += epoch_time

        print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | Total elapsed: {elapsed_time:.2f}s")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch+1, model_num, elapsed_time, avg_train_loss, avg_val_loss, val_accuracy)

    total_training_time = elapsed_time
    shared_training_loss[model_num] = avg_train_loss
    shared_validation_loss[model_num] = avg_val_loss
    shared_acc[model_num] = val_accuracy
    shared_time[model_num] = total_training_time

def predict_next_char(model, char_to_ix, ix_to_char, initial_str, max_length):
    model.eval()
    with torch.no_grad():
        # Prepare the source input (src)
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0).to(device)
        
        # Start the target sequence (tgt) with the SOS token
        tgt_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        for _ in range(max_length):
            # Generate the next token
            output = model(initial_input, tgt_input)
            next_token = torch.argmax(output, dim=1).unsqueeze(0)

            # Append the predicted token to the target sequence
            tgt_input = torch.cat([tgt_input, next_token], dim=1)

            # Stop if the EOS token is predicted
            if next_token.item() == EOS_token:
                break

        # Return the predicted character
        predicted_index = tgt_input[0, -1].item()
        return ix_to_char[predicted_index]

learning_rate = 0.001
epochs = 100
batch_size = 1024
hidden_size = [128]
fc_layers = [1, 2, 4]
nhead = [2, 4]


if __name__ == '__main__':

    with open('tinyShakeSpeare.txt', 'r') as f:
        text = f.read()

    model_Iter = ["Transformer"]
    max_length = [20, 30, 50]
    number_of_models = len(hidden_size)*len(fc_layers)*len(model_Iter)*len(max_length)*len(nhead)

    process_num = -1
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

    training_loss_results = [None] * number_of_models
    validation_loss_results = [None] * number_of_models
    validation_accuracy_results = [None] * number_of_models
    training_time_results = [None] * number_of_models
    inference_time_results  = []
    model_parameters_results = [None] * number_of_models
    model_size_results = [None] * number_of_models

    for i in max_length:
        length_num += 1
        train_dataset_list.append(ShakespeareDataset(text, char_to_ix, i))
        train_loader_list.append(DataLoader(train_dataset_list[length_num], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2))
        val_loader_list.append(DataLoader(train_dataset_list[length_num], batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2))

    for i in hidden_size:
        for j in fc_layers:
            for k in range(len(max_length)):
                for l in nhead:
                    models.append(CharTransformer(len(chars), i, len(chars), j, l).to(device))
                    model_length.append(max_length[k])
                    process_num += 1
                    train_model(models[process_num], train_loader_list[k], val_loader_list[k], process_num, training_loss_results, validation_loss_results, validation_accuracy_results, training_time_results)

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
        max_sequence_length = 1000

        for _ in range(max_sequence_length):
            predicted_char = predict_next_char(i, char_to_ix, ix_to_char, generated_sequence, model_length[model_num])
            generated_sequence += predicted_char
        
        inference_time = time.time() - start_time
        inference_time_results.append(inference_time)
        predicted_char_results.append(generated_sequence)

        model_size(i, model_parameters_results, model_size_results, model_num)
        print(f"Model {model_num+1}, Transformer, Sequence Length {model_length[model_num]}, Training Loss: {training_loss_results[model_num]}, Validation Loss: {validation_loss_results[model_num]}, Validation Accuracy: {validation_accuracy_results[model_num]}, Training time: {training_time_results[model_num]}, Inference time:  {inference_time_results[model_num]}, Hidden Sizes: {i.get_hidden_size()}, Fully Connected Layers: {i.get_fc_layers()}, Heads : {i.get_nhead()}, Parameters: {model_parameters_results[model_num]}, Size: {model_size_results[model_num]} MB, Predicted next character: '{predicted_char_results[model_num]}'")
        model_data = {
            "Model": model_num + 1,
            "Transformer": "Transformer",
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
        "Model", "Transformer", "Sequence Length", "Training Loss", "Validation Loss", "Validation Accuracy", 
        "Training time", "Inference Time", "Hidden Size", "Fully Connected Layers", "Parameters", "Size (MB)", "Predicted next character"
    ]

    with open('model_results_spear.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_data)

