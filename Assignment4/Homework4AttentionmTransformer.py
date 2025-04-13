import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=False
mp.set_start_method('spawn', force=True) 
csvFiles = ["Dataset - English to French.csv", "Dataset - French to English.csv"]
SOS_token = 0
EOS_token = 1
max_length = 100 # Must be 100?
hidden_size = 128 # Adjustable
learning_rate = 0.01 # Adjustable
n_epochs = 41 # Adjustable
batch_size = 1 # Must be 1?
num_layers = [1, 2, 4] # Adjustable
nhead = [2, 4]

class VocabularyWeights(Dataset):
    """Custom Dataset class for handling synonym pairs."""
    def __init__(self, dataset, word_to_index):
        self.dataset = dataset
        self.word_to_index = word_to_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_sentence, target_sentence = self.dataset[idx]
        input_tensor = torch.tensor([self.word_to_index[word] for word in input_sentence.split()] + [EOS_token], dtype=torch.long)
        target_tensor = torch.tensor([self.word_to_index[word] for word in target_sentence.split()] + [EOS_token], dtype=torch.long)
        return input_tensor, target_tensor
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))  # (d_model//2)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]  # Add positional encoding
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(self.hidden_size)  # (seq_len, batch, hidden_size)
        embedded = self.pos_encoder(embedded)
        memory = self.transformer_encoder(embedded)
        return memory  # This replaces encoder_outputs

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, nhead, max_length=12, dropout_p=0.1):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.pos_decoder = PositionalEncoding(hidden_size, dropout=dropout_p)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, tgt, memory):
        # tgt: (tgt_seq_len, batch)
        # memory: (src_seq_len, batch, hidden_size)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.hidden_size)
        tgt_emb = self.pos_decoder(tgt_emb)

        output = self.transformer_decoder(tgt_emb, memory)
        output = self.out(output)
        return output

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, word_to_index, num_layers, nhead):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = input_tensor.transpose(0, 1)  # (seq_len, batch)
    target_tensor = target_tensor.transpose(0, 1)  # (seq_len, batch)

    memory = encoder(input_tensor)  # (src_seq_len, batch, hidden_size)

    tgt_input = target_tensor[:-1, :]  # exclude EOS
    tgt_output = target_tensor[1:, :]  # predict next tokens

    output = decoder(tgt_input, memory)  # (tgt_seq_len - 1, batch, vocab_size)

    output = output.view(-1, output.size(-1))
    tgt_output = tgt_output.contiguous().view(-1)

    loss = criterion(torch.log_softmax(output, dim=1), tgt_output)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def evaluate(encoder, decoder, dataloader, criterion, index_to_word, num_layers, nhead):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            input_tensor = input_tensor.to(device).transpose(0, 1)
            target_tensor = target_tensor.to(device).transpose(0, 1)

            memory = encoder(input_tensor)

            tgt_input = target_tensor[:-1, :]
            tgt_output = target_tensor[1:, :]

            output = decoder(tgt_input, memory)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(torch.log_softmax(output, dim=1), tgt_output)
            total_loss += loss.item()

            predictions = output.argmax(dim=1)
            correct += (predictions == tgt_output).sum().item()
            total += tgt_output.size(0)

    encoder.train()
    decoder.train()

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy

def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, index_to_word, num_layers, nhead):
    encoder.eval()
    decoder.eval()
    print("\nSample Translations:\n")

    with torch.no_grad():
        for idx, (input_tensor, target_tensor) in enumerate(dataloader):
            if idx == 5:
                break

            input_tensor = input_tensor.to(device).transpose(0, 1)
            target_tensor = target_tensor.to(device).transpose(0, 1)

            memory = encoder(input_tensor)

            tgt_input = torch.tensor([[SOS_token]], dtype=torch.long).to(device)  # Start of sentence
            outputs = []

            for _ in range(max_length):
                output = decoder(tgt_input, memory)
                next_token = output[-1].argmax(dim=-1)
                outputs.append(next_token.item())

                if next_token.item() == EOS_token:
                    break

                tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=0)

            input_words = [index_to_word[idx.item()] for idx in input_tensor[:, 0] if idx.item() in index_to_word and idx.item() != EOS_token]
            target_words = [index_to_word[idx.item()] for idx in target_tensor[:, 0] if idx.item() in index_to_word and idx.item() != EOS_token]
            predicted_words = [index_to_word[idx] for idx in outputs if idx in index_to_word and idx != EOS_token]

            print(f"Input:     {' '.join(input_words)}")
            print(f"Target:    {' '.join(target_words)}")
            print(f"Predicted: {' '.join(predicted_words)}")
            print()
        
def train_model(fileName, hidden_size, learning_rate, n_epochs, batch_size, SOS_token, EOS_token, max_length, num_layers, nhead):

    with open(fileName, encoding="utf-8") as file:
        reader = csv.reader(file)
        Dataset = [tuple(row) for row in reader]

    words = set(word for pair in Dataset for sentence in pair for word in sentence.split())
    word_to_index = {"SOS": SOS_token, "EOS": EOS_token, **{word: i+2 for i, word in enumerate(sorted(words))}}
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    vocabulary_weights = VocabularyWeights(Dataset, word_to_index)
    dataloader = DataLoader(vocabulary_weights, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    encoder = TransformerEncoder(input_size=len(word_to_index), hidden_size=hidden_size, num_layers = num_layers, nhead = nhead).to(device)
    decoder = TransformerDecoder(hidden_size=hidden_size, output_size=len(word_to_index), num_layers = num_layers, nhead = nhead).to(device)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    training_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(n_epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, word_to_index, num_layers, nhead)
            total_loss += loss
        
        training_loss = total_loss / len(dataloader)
        training_losses.append(training_loss)

        if epoch % 10 == 0:
            val_loss, val_accuracy = evaluate(encoder, decoder, dataloader, criterion, index_to_word, num_layers, nhead)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            print(f'Epoch {epoch}, Training Loss: {training_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

    evaluate_and_show_examples(encoder, decoder, dataloader, criterion, index_to_word, num_layers, nhead)

    epochs = range(0, n_epochs, 10)
    plt.figure()
    plt.plot(epochs, training_losses[::10], label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(f'{fileName}_loss_{num_layers}_{nhead}.png')

    plt.figure()
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.savefig(f'{fileName}_accuracy_{num_layers}_{nhead}.png')

def collate_fn(batch):
    input_tensors, target_tensors = zip(*batch)
    input_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=EOS_token)
    target_tensors = pad_sequence(target_tensors, batch_first=True, padding_value=EOS_token)
    return input_tensors, target_tensors

if __name__ == '__main__':
    for i in csvFiles:
        for j in num_layers:
            for k in nhead:
                train_model(i, hidden_size, learning_rate, n_epochs, batch_size, SOS_token, EOS_token, max_length, j, k)
