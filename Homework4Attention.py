import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=False
mp.set_start_method('spawn', force=True) 
csvFiles = ["Dataset - English to French.csv", "Dataset - French to English.csv"]
SOS_token = 0
EOS_token = 1
max_length = 100 # Must be 100?
hidden_size = 256
learning_rate = 0.01
n_epochs = 41
batch_size = 1 # Must be 1?
attention = [False, True]

class SynonymDataset(Dataset):
    """Custom Dataset class for handling synonym pairs."""
    def __init__(self, dataset, char_to_index):
        self.dataset = dataset
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_word, target_word = self.dataset[idx]
        input_tensor = torch.tensor([self.char_to_index[char] for char in input_word] + [EOS_token], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_index[char] for char in target_word] + [EOS_token], dtype=torch.long)
        return input_tensor, target_tensor
    
class Encoder(nn.Module):
    """The Encoder part of the seq2seq model."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.GRU(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device))
    
class Decoder(nn.Module):
    """The Decoder part of the seq2seq model."""
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
                             
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.GRU(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device))
    
class AttnDecoder(nn.Module):
    """Decoder with attention mechanism."""
    def __init__(self, hidden_size, output_size, max_length=12, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.GRU = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0].unsqueeze(0)), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.GRU(output, hidden)

        output = torch.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device))

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, char_to_index, attention, max_length = 100):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device).to(device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[char_to_index['SOS']]], device=device).to(device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        if attention: decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        else: decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == char_to_index['EOS']:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, index_to_char, attention, n_examples=5):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)
            
            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device).to(device)

            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                if attention: decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                else: decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            total_loss += loss.item() / target_length
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            if i < n_examples:
                predicted_string = ''.join([index_to_char[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_string = ''.join([index_to_char[index.item()] for index in target_tensor if index.item() not in (SOS_token, EOS_token)])
                input_string = ''.join([index_to_char[index.item()] for index in input_tensor if index.item() not in (SOS_token, EOS_token)])
                
                print(f'Input: {input_string}, Target: {target_string}, Predicted: {predicted_string}')
        
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')
        
def train_model(fileName, hidden_size, learning_rate, n_epochs, batch_size, SOS_token, EOS_token, attention, max_length):

    with open (fileName, encoding="utf-8") as file:
        reader = csv.reader(file)
        Dataset = [tuple(row) for row in reader]
    
    char_to_index = {"SOS": SOS_token, "EOS": EOS_token, **{char: i+2 for i, char in enumerate(sorted(list(set(''.join([word for pair in Dataset for word in pair])))))}}
    index_to_char = {i: char for char, i in char_to_index.items()}
    synonym_dataset = SynonymDataset(Dataset, char_to_index)
    dataloader = DataLoader(synonym_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    encoder = Encoder(input_size = len(char_to_index), hidden_size=hidden_size).to(device)
    if attention: decoder = AttnDecoder(hidden_size=hidden_size, output_size = len(char_to_index), max_length = max_length).to(device)
    else: decoder = Decoder(hidden_size=hidden_size, output_size = len(char_to_index)).to(device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(n_epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)
            
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, char_to_index, attention)
            total_loss += loss
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')

    evaluate_and_show_examples(encoder, decoder, dataloader, criterion, index_to_char, attention)

if __name__ == '__main__':
    for i in csvFiles:
        for j in attention:
            train_model(i, hidden_size, learning_rate, n_epochs, batch_size, SOS_token, EOS_token, j, max_length)