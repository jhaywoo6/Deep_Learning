import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from torch.utils.data import random_split
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split
import csv
from thop import profile
from transformers import SwinForImageClassification, SwinConfig


class SwinEmbedding(nn.Module):
    def __init__(self, patch_size=4, emb_size=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, emb_size, kernel_size = patch_size, stride = patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        
    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.rearrange(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.linear = nn.Linear(4*emb_size, 2*emb_size)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L)/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)
        x = self.linear(x)
        return x
    
class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3*emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1
    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        height = width = int(np.sqrt(x.shape[1]))
        x = self.linear1(x)
        
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        
        if self.shifted:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))
        
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)            
        
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)
        
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding
        
        if self.shifted:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask
        
        wei = F.softmax(wei, dim=-1) @ V
        
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        return self.linear2(x)
    
class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
                         nn.Linear(emb_size, 4*emb_size),
                         nn.GELU(),
                         nn.Linear(4*emb_size, emb_size),
                  )
    
    def forward(self, x):
        return self.ff(x)
    
class SwinEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7):
        super().__init__()
        self.WMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=False)
        self.SWMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=True)
        self.ln = nn.LayerNorm(emb_size)
        self.MLP = MLP(emb_size)
        
    def forward(self, x):
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        return x
    
class Swin(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = SwinEmbedding()
        self.PatchMerging = nn.ModuleList()
        emb_size = 96
        num_class = 100
        for i in range(3):
            self.PatchMerging.append(PatchMerging(emb_size))
            emb_size *= 2
        
        self.stage1 = SwinEncoder(96, 3)
        self.stage2 = SwinEncoder(192, 6)
        self.stage3 = nn.ModuleList([SwinEncoder(384, 12),
                                     SwinEncoder(384, 12),
                                     SwinEncoder(384, 12) 
                                    ])
        self.stage4 = SwinEncoder(768, 24)
        
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=49)
        
        self.layer = nn.Linear(768, num_class)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.stage1(x)
        x = self.PatchMerging[0](x)
        x = self.stage2(x)
        x = self.PatchMerging[1](x)
        for stage in self.stage3:
            x = stage(x)
        x = self.PatchMerging[2](x)
        x = self.stage4(x)
        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))
        return x
    
def train_model(model, epochs, isPreTrained, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if isPreTrained:
                outputs = model(pixel_values=inputs)
                logits = outputs.logits
                loss = criterion(logits, labels) / params['accum_steps']
            else:
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
                if isPreTrained:
                    outputs = model(pixel_values=inputs)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                val_loss += loss.item()

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
            if isPreTrained:
                outputs = model(pixel_values=inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
            test_loss += loss.item()
            
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

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    params = {
        'patch_size' : 8,
        'embed_dim' : 512,
        'layers' : 8,
        'num_heads' : 2,
        'pretrained_learning_rate' : 2*10**-5,
        'scratch_learning_rate' : 0.001,
        'batch_size' : 32, # Gradient Accumulatuion used to multiply this value to reach a larger batch size
        'SWIN_epochs' : 5,
        'accum_steps' : 1,
        'classes' : 100
    }

    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    results = {
        'models' : [],
        'avg_times' : [],
        'num_params' : [],
        'flops' : [],
        'accuracy' : []
    }

    scratch_model = Swin().to(device)

    tiny_model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=1000
    )

    tiny_model.classifier = nn.Linear(tiny_model.classifier.in_features, params['classes'])
    tiny_model = tiny_model.to(device)

    small_model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-small-patch4-window7-224",
        num_labels=1000
    )

    small_model.classifier = nn.Linear(small_model.classifier.in_features, params['classes'])
    small_model = small_model.to(device)

    def freeze_backbone(model):
        for name, param in model.named_parameters():
            if "classifier" not in name:  # Only allow classifier (head) to train
                param.requires_grad = False

    freeze_backbone(tiny_model)
    freeze_backbone(small_model)
    
    SwinModels = [[tiny_model, True, params['pretrained_learning_rate']], [small_model, True, params['pretrained_learning_rate']], [scratch_model, False, params['scratch_learning_rate']]]

    for i in SwinModels:
        train_model(i[0], params["SWIN_epochs"], i[1], i[2])

    headers = ['Model', 'Average Time (s)', 'Num Parameters', 'FLOPs', 'Accuracy (%)']

    model_names = ['Tiny', 'Small', 'Scratch']

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