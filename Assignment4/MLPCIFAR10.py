import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class multiLayerPerceptronA(nn.Module):
    def __init__(self, inputSize, hiddenSizes, outputSize):
        super(multiLayerPerceptronA, self).__init__()
        layers = []
        prevSize = inputSize
        
        for hiddenSize in hiddenSizes:
            layers.append(nn.Linear(prevSize, hiddenSize))
            layers.append(nn.ReLU())
            prevSize = hiddenSize

        layers.append(nn.Linear(prevSize, outputSize))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class multiLayerPerceptronB(nn.Module):
    def __init__(self, inputSize, hiddenSizes, outputSize):
        super(multiLayerPerceptronB, self).__init__()
        layers = []
        prevSize = inputSize
        
        for hiddenSize in hiddenSizes:
            layers.append(nn.Linear(prevSize, hiddenSize))
            layers.append(nn.ReLU())
            prevSize = hiddenSize

        layers.append(nn.Linear(prevSize, outputSize))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



def evaluate_accuracy(model, dataLoader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in dataLoader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return 100 * correct / total

def trainLoopA(epochs, trainLoader, device, model, optimizer, modelNum, criterion, valLoader, queue):
    trainLosses = []
    trainAccuracies = []
    valAccuracies = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        runningLoss = 0.0

        for batchIdx, (data, targets) in enumerate(trainLoader):
            data = data.to(device, dtype=torch.float32)
            targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            _, predictions = torch.max(scores, 1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        trainLosses.append(runningLoss / len(trainLoader))
        trainAccuracies.append(100 * correct / total)
        valAccuracy = evaluate_accuracy(model, valLoader, device)
        valAccuracies.append(valAccuracy)

        print(f"Model {modelNum} Epoch [{epoch+1}/{epochs}], Loss: {trainLosses[-1]:.4f}, Train Accuracy: {trainAccuracies[-1]:.2f}%, Val Accuracy: {valAccuracy:.2f}%")
    
    queue.put((trainLosses, trainAccuracies, valAccuracies))

def trainLoopB(epochs, trainLoader, device, model, optimizer, modelNum, criterion, valLoader, queue):
    trainLosses = []
    trainAccuracies = []
    valAccuracies = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        runningLoss = 0.0

        for batchIdx, (data, targets) in enumerate(trainLoader):
            data = data.to(device, dtype=torch.float32)
            targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            correct += (scores == targets).sum().item()
            total += targets.size(0)

        trainLosses.append(runningLoss / len(trainLoader))
        trainAccuracies.append(100 * correct / total)
        valAccuracy = evaluate_accuracy(model, valLoader, device)
        valAccuracies.append(valAccuracy)

        print(f"Model {modelNum} Epoch [{epoch+1}/{epochs}], Loss: {trainLosses[-1]:.4f}, Train Accuracy: {trainAccuracies[-1]:.2f}%, Val Accuracy: {valAccuracy:.2f}%")
    
    queue.put((trainLosses, trainAccuracies, valAccuracies))

def trainModels(modelNum, argumentsSet):
    mp.set_start_method("spawn", force=True)
    processes = []
    queue = mp.Queue()

    for i in range(modelNum):
        args = argumentsSet[i] + (queue,)
        if i >= 2:
            p = mp.Process(target=trainLoopB, args=args)
        else:
            p = mp.Process(target=trainLoopA, args=args)
        p.start()
        processes.append(p)

    results = []
    for p in processes:
        p.join()
        results.append(queue.get())

    return results
    
def evaluateA(model, testLoader):
    model.eval()
    allPredictions = []
    allLabels = []

    with torch.no_grad():
        for data, targets in testLoader:
            data = data.to(device, dtype=torch.float32)
            targets.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            allPredictions.extend(predictions.cpu().numpy())
            allLabels.extend(targets.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(allLabels, allPredictions, average='weighted')
    confusionMatrix = confusion_matrix(allLabels, allPredictions)

    print("\nResults:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusionMatrix)

def evaluateB(model, testLoader):
    model.eval()
    allPredictions = []
    allLabels = []

    with torch.no_grad():
        for data, targets in testLoader:
            data = data.to(device, dtype=torch.float32)
            targets.to(device, dtype=torch.float32)
            outputs = model(data).cpu().numpy()
            allPredictions.extend(outputs.flatten())
            allLabels.extend(targets.cpu().numpy())
    
    mse = mean_squared_error(allLabels, allPredictions)
    mae = mean_absolute_error(allLabels, allPredictions)
    rmse = mse ** 0.5

    print("\nResults:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

def plotResults(trainLosses, trainAccuracies, valAccuracies, label, figure):
    epochs = range(1, len(trainLosses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(figure, 2, 1)
    plt.plot(epochs, trainLosses, label=f'{label} Loss', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    
    plt.subplot(figure, 2, 2)
    plt.plot(epochs, trainAccuracies, label=f'{label} Train Accuracy', marker='o')
    plt.plot(epochs, valAccuracies, label=f'{label} Validation Accuracy', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.draw()
    plt.pause(0.001)

def select_columns(data, selectedFeatures):
    return data.iloc[:, selectedFeatures]

    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []
    modelNum = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainDatasetCIFAR = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    testDatasetCIFAR = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    data = pd.read_csv("Housing.csv")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_features = encoder.fit_transform(data.iloc[:, [12]])
    encoded_features = encoded_features.astype(int)
    encoded_df = pd.DataFrame(encoded_features, columns=['Feature 13', 'Feature 14', 'Feature 15'])
    data = pd.concat([data, encoded_df], axis=1)
    data.to_csv("Housing_with_encoded.csv", index=False)        
    data = pd.read_csv('Housing_with_encoded.csv')
    yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    data[yes_no_columns] = data[yes_no_columns].map(lambda x: 1 if x.lower() == 'yes' else 0)
    furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    data['furnishingstatus'] = data['furnishingstatus'].map(furnishing_map)
    scaler_X = StandardScaler()

    # Model 1 for problem 1a. 3 Hidden Layers, 20 epoches.

    inputSize_1 = 32 * 32 * 3
    hiddenSizes_1 = [64, 32, 64]
    outputSize_1 = 10
    learningRate_1 = 0.001
    epochs_1 = 20
    batchSize_1 = 16

    model_1 = multiLayerPerceptronA(inputSize_1, hiddenSizes_1, outputSize_1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(model_1.parameters(), lr=learningRate_1)
    trainLoader_1 = DataLoader(trainDatasetCIFAR, batch_size=batchSize_1, shuffle=True)
    testLoader_1 = DataLoader(testDatasetCIFAR, batch_size=batchSize_1, shuffle=False)
    valLoader_1 = torch.utils.data.DataLoader(testDatasetCIFAR, batch_size=batchSize_1, shuffle=False)

    arguments_1 = (epochs_1, trainLoader_1, device, model_1, optimizer_1, 1, criterion, valLoader_1)

    # Model 2 for problem 1b. 5 Hidden Layers, 20 epoches.

    inputSize_2 = 32 * 32 * 3
    hiddenSizes_2 = [512, 256, 128, 256, 512]
    outputSize_2 = 10
    learningRate_2 = 0.001
    epochs_2 = 20
    batchSize_2 = 16

    model_2 = multiLayerPerceptronA(inputSize_2, hiddenSizes_2, outputSize_2).to(device)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=learningRate_2)
    trainLoader_2 = DataLoader(trainDatasetCIFAR, batch_size=batchSize_2, shuffle=True)
    testLoader_2 = DataLoader(testDatasetCIFAR, batch_size=batchSize_2, shuffle=False)
    valLoader_2 = torch.utils.data.DataLoader(testDatasetCIFAR, batch_size=batchSize_2, shuffle=False)

    arguments_2 = (epochs_2, trainLoader_2, device, model_2, optimizer_2, 2, criterion, valLoader_2)

    # Model 3 for problem 2a. 7 input features. Column 12 treated as quantitative.

    inputSize_3 = 7
    hiddenSizes_3 = [64, 32, 64]
    outputSize_3 = 1
    learningRate_3 = 0.0001
    epochs_3 = 20
    batchSize_3 = 16

    selectedFeatures_3 = [1, 2, 3, 4, 7, 9, 12]
    X = select_columns(data, selectedFeatures_3)
    y = data['price']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    trainLoaderHousing_3 = TensorDataset(X_train_tensor, y_train_tensor)
    testLoaderHousing_3 = TensorDataset(X_test_tensor, y_test_tensor)
    valLoader_3 = TensorDataset(X_val_tensor, y_val_tensor)

    model_3 = multiLayerPerceptronB(inputSize_3, hiddenSizes_3, outputSize_3).to(device)
    criterion = nn.MSELoss()
    optimizer_3 = optim.Adam(model_3.parameters(), lr=learningRate_3)
    valLoader_3 = torch.utils.data.DataLoader(valLoader_3, batch_size=batchSize_3, shuffle=False)

    arguments_3 = (epochs_3, trainLoaderHousing_3, device, model_3, optimizer_3, 3, criterion, valLoader_3)

    # Model 4 for problem 2b. 7 input features. Column 12 split into 3 features using on hot encoding.

    inputSize_4 = 7
    hiddenSizes_4 = [64, 32, 64]
    outputSize_4 = 1
    learningRate_4 = 0.0001
    epochs_4 = 20
    batchSize_4 = 16

    selectedFeatures_4 = [1, 2, 3, 4, 13, 14, 15]
    X = select_columns(data, selectedFeatures_4)
    y = data['price']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    trainLoaderHousing_4 = TensorDataset(X_train_tensor, y_train_tensor)
    testLoaderHousing_4 = TensorDataset(X_test_tensor, y_test_tensor)
    valLoader_4 = TensorDataset(X_val_tensor, y_val_tensor)

    model_4 = multiLayerPerceptronB(inputSize_4, hiddenSizes_4, outputSize_4).to(device)
    criterion = nn.MSELoss()
    optimizer_4 = optim.Adam(model_4.parameters(), lr=learningRate_4)
    valLoader_4 = torch.utils.data.DataLoader(valLoader_4, batch_size=batchSize_4, shuffle=False)

    arguments_4 = (epochs_4, trainLoaderHousing_4, device, model_4, optimizer_4, 4, criterion, valLoader_4)

    # Model 5 for problem 2c. 14 input features. Column 12 split into 4 features using on hot encoding. Model width and length increased.
    
    inputSize_5 = 14
    hiddenSizes_5 = [128, 64, 32, 64, 128]
    outputSize_5 = 1
    learningRate_5 = 0.0001
    epochs_5 = 20
    batchSize_5 = 16
    
    selectedFeatures_5 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]
    X = select_columns(data, selectedFeatures_5)
    y = data['price']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    trainLoaderHousing_5 = TensorDataset(X_train_tensor, y_train_tensor)
    testLoaderHousing_5 = TensorDataset(X_test_tensor, y_test_tensor)
    valLoader_5 = TensorDataset(X_val_tensor, y_val_tensor)

    model_5 = multiLayerPerceptronB(inputSize_5, hiddenSizes_5, outputSize_5).to(device)
    criterion = nn.MSELoss()
    optimizer_5 = optim.Adam(model_5.parameters(), lr=learningRate_5)
    valLoader_5 = torch.utils.data.DataLoader(valLoader_5, batch_size=batchSize_5, shuffle=False)

    arguments_5 = (epochs_5, trainLoaderHousing_5, device, model_5, optimizer_5, 5, criterion, valLoader_5)

    argumentsSet = [arguments_1, arguments_2, arguments_3, arguments_4, arguments_5]

    results = trainModels(5, argumentsSet)

    plotResults(*results[0], "Model 1", 1)
    evaluateA(model_1, testLoader_1)
    torch.save(model_1, 'model_1.pth')
    plotResults(*results[1], "Model 2", 2)
    evaluateA(model_2, testLoader_2)
    torch.save(model_2, 'model_2.pth')
    
    plotResults(*results[2], "Model 3", 3)
    evaluateB(model_3, testLoaderHousing_3)
    torch.save(model_3, 'model_3.pth')
    plotResults(*results[3], "Model 4", 4)
    evaluateB(model_4, testLoaderHousing_4)
    torch.save(model_4, 'model_4.pth')
    plotResults(*results[4], "Model 5", 5)
    evaluateB(model_5, testLoaderHousing_5)
    torch.save(model_5, 'model_5.pth')
    

    plt.show(block=False)
    plt.pause(0.001)

    input("Press Enter to exit...")
