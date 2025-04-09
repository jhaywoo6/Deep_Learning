import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import itertools
import random
import time
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split
import csv
from thop import profile
import timm

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))  

    def forward(self, x):
        return x + self.pos_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=100, embed_dim=768, num_heads=8, layers=6, mlp_dim=1024):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.mlp_head(x[:, 0])

def train_model(model, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / params['accum_steps']
            scaler.scale(loss).backward()
            if (batch_idx + 1) % params['accum_steps'] == 0 or (batch_idx + 1) == total_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * params['accum_steps']

            percent_complete = 100 * (batch_idx + 1) / total_batches
            print(f"Epoch {epoch+1} Progress: {percent_complete:.2f}% ({batch_idx + 1}/{total_batches}), {time.time()-start_time:.2f}s", end='\r')
        
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_preds)
        epoch_times.append(time.time() - start_time)
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy*100:.2f}%, "
              f"Time: {epoch_times[-1]:.2f}s")

    model.eval()
    test_preds, test_targets = [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    x = torch.randn([params['batch_size'], 3, 224, 224]).to(device)
    flops, _ = profile(model, inputs=(x,))
    test_loss /= len(test_loader)
    test_accuracy = accuracy_score(test_targets, test_preds)
    results['models'].append(model)
    results['avg_times'].append(sum(epoch_times) / len(epoch_times))
    results['num_params'].append(sum(p.numel() for p in model.parameters()))
    results['flops'].append(flops)
    results['accuracy'].append(test_accuracy)

    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%, Time Per Epoch: {results['avg_times'][-1]:.2f}, Number of Parameters: {results['num_params'][-1]}, FLOPs: {results['flops'][-1]}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

params = {
    'patch_size' : [4, 8],
    'embed_dim' : [512],
    'layers' : [4, 8],
    'num_heads' : [2, 4],
    'learning_rate' : 0.001,
    'batch_size' : 32, # Gradient Accumulatuion used to multiply this value to reach a larger batch size
    'ViT_epochs' : random.randint(20, 50),
    'ResNet_epochs' : 10,
    'accum_steps' : 2
}

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

all_combinations = list(itertools.product(
    params['patch_size'],
    params['embed_dim'],
    params['layers'],
    params['num_heads']
))

selected_combinations = random.sample(all_combinations, 4)
results = {
    'models' : [],
    'avg_times' : [],
    'num_params' : [],
    'flops' : [],
    'accuracy' : []
}

for idx, (i, j, k, l) in enumerate(selected_combinations):
    model = VisionTransformer(patch_size = i, embed_dim = j, layers = k, num_heads = l).to(device)
    print(f"Model {idx + 1}, Patch Size: {i}, Embed Dim: {j}, Layers: {k}, Num Heads: {l}, Epochs: {params['ViT_epochs']}")
    train_model(model, params['ViT_epochs'])

resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)

resnet18 = resnet18.to(device)

train_model(resnet18, params['ResNet_epochs'])
    
headers = ['Model', 'Average Time (s)', 'Num Parameters', 'FLOPs', 'Accuracy (%)']

model_names = ['ViT-A', 'ViT-B', 'ViT-C', 'ViT-D', 'ResNet-18']

rows = []

for i, model in enumerate(results['models']):
    avg_time = results['avg_times'][i]
    num_params = results['num_params'][i]
    flops = results['flops'][i]
    accuracy = results['accuracy'][i]
    
    rows.append([model_names[i], avg_time, num_params, flops, accuracy])

csv_file = 'HW6_ViT_RES18_results.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(rows)

