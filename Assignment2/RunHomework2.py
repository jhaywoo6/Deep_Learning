import multiprocessing
import subprocess
import time
import os

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

scripts = [
    "C:/Users/Jacob/Assignment2DL/Model/TrainAlexNetCIFAR10D.py",
    "C:/Users/Jacob/Assignment2DL/Model/TrainAlexNetCIFAR100D.py",
    "C:/Users/Jacob/VGG11CIFAR10D.py",
    "C:/Users/Jacob/VGG11CIFAR100D.py",
    "C:/Users/Jacob/Assignment2DL/Model/TrainAlexNetCIFAR10ND.py",
    "C:/Users/Jacob/Assignment2DL/Model/TrainAlexNetCIFAR100ND.py",
    "C:/Users/Jacob/VGG11CIFAR10ND.py",
    "C:/Users/Jacob/VGG11CIFAR100ND.py",
    "C:/Users/Jacob/ResNet11CIFAR10D.py",
    "C:/Users/Jacob/ResNet11CIFAR100D.py",
    "C:/Users/Jacob/ResNet11CIFAR10ND.py",
    "C:/Users/Jacob/ResNet11CIFAR100ND.py",
    "C:/Users/Jacob/ResNet18CIFAR10D.py",
    "C:/Users/Jacob/ResNet18CIFAR100D.py",
    "C:/Users/Jacob/ResNet18CIFAR10ND.py",
    "C:/Users/Jacob/ResNet18CIFAR100ND.py"
]

def run_script(script_path):
    subprocess.run(["python", script_path])

def run_batch(batch):
    processes = []
    for script in batch:
        p = multiprocessing.Process(target=run_script, args=(script,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    batch_size = 4

    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i+batch_size]
        print(f"Running batch: {batch}")
        run_batch(batch)
        time.sleep(1)

    print("All scripts have finished.")
