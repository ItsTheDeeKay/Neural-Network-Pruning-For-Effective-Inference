# Here we import all the necessary libraries...

import os
import time
import torch
import numpy as np
import seaborn as sb
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# # # # # # # # # # # # # # # # #  CSE 676 Deep Learning Bonus Task  # # # # # # ## # # # # # # # # # 

# This will load the dataset from the same .ipynb directory... 
imagenet_Dataset = ImageFolder("/projects/academic/courses/cse676smr23/deekayG/smallfiles_dontworry/10_Classes/")
directory = "/projects/academic/courses/cse676smr23/deekayG/smallfiles_dontworry/10_Classes/"
resize = transforms.Resize((224, 224))

# This will normalize the input dataset to have zero mean and unit variance...
normalized_Samples = transforms.Normalize(mean = [0.47, 0.45, 0.39], std = [0.23, 0.23, 0.23])
tensors_Samples = transforms.ToTensor()
transformed = transforms.Compose([resize, tensors_Samples, normalized_Samples])

# Now we have pre-processed dataset ready to be splitted and training...
proccessed_Dataset = ImageFolder("/projects/academic/courses/cse676smr23/deekayG/smallfiles_dontworry/10_Classes/", transform = transformed)

# This will split the dataset into training and validation sets...
train_70 = int((70/100) * (len(imagenet_Dataset)))
val_15 = int((15/100) * (len(imagenet_Dataset)))
test_X = len(imagenet_Dataset) - train_70 - val_15
train_Dataset, val_Dataset, test_Dataset = random_split(proccessed_Dataset, [train_70 , val_15, test_X])

# This will create PyTorch DataLoader objects to be later loaded into NN architecture...
train_Dataloader = DataLoader(train_Dataset, batch_size = 64, shuffle = True)
val_Dataloader = DataLoader(val_Dataset, batch_size = 64, shuffle = False)
test_Dataloader = DataLoader(test_Dataset, batch_size = 64, shuffle = False)

# This will count the no. classes in the dataset...
def count_Classes(directory):
    count = 0
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            count += 1
    return count

tot_Cls = count_Classes(directory)
class_Names = test_Dataloader.dataset.dataset.classes

# This will print the insights of our dataset...
for i, j in test_Dataloader:
    print("Total number of images:", len(imagenet_Dataset))
    print(f"Number of classes in the dataset: {tot_Cls}")
    print("Total number of images to be trained:", len(train_Dataset))
    print("Total number of images for validation:", len(val_Dataset))
    print("Total number of images to be tested:", len(test_Dataset))
    print(f"Number of images in the batch (batch size): {i.shape[0]}")
    print(f"Number of channels in each image: {i.shape[1]}")
    print(f"Dimensions of each image: {i.shape[2], i.shape[3]}")
    
    break

# # # # # # # # # # # # # # # # # # # # # # # VGG16bn Model # # # # # # # # # # # # # # # # # # # # # # 

# This will check if GPU harnessing is available otherwise it will assign cpu as training resource...
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")

# This will start training the model upon calling...
def train(dataloader, vgg_Mo, loss_Func, optimizer):
    
    size = len(dataloader.dataset)
    vgg_Mo.train()
    train_Loss = 0
    train_Correct = 0
    train_Total = 0
    for batch, (i, j) in enumerate(dataloader):
        pr, ls = i.to(device), j.to(device)

        # This will compute prediction error...
        pred = vgg_Mo(pr)
        loss = loss_Func(pred, ls)

        # This will perform the backpropagation to optimize the network with optimal weights...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_Loss += loss.item()
        train_Total += ls.size(0)
        train_Correct += (torch.argmax(pred, dim = 1) == j.to(device)).sum().item()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(pr)
            print(f"Training Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#     print(pred.shape)
#     print(ls.shape)
    train_Loss /= size
    train_Accuracy = 100 * (train_Correct / train_Total)
#     print(f"Train Accuracy: {(train_Accuracy):>0.1f}%, Avg Training Loss: {train_Loss:>8f} \n")

    return train_Loss, train_Accuracy

def prune_model(model, amount):
    """
    Function to prune the model's convolutional layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

def validate(dataloader, vgg_Mo, loss_fn):
    size = len(dataloader.dataset)
    vgg_Mo.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = vgg_Mo(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     print(pred.shape)
#     print(y.shape)

    val_loss /= size
    val_accuracy = 100* (correct / size)
#     print(f"Validation Accuracy: {(val_accuracy):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    return val_loss, val_accuracy

# This function is perform the testing by the trained model over new data...
def test(dataloader, vgg_Mo, loss_Func):
    
    size = len(dataloader.dataset)
    vgg_Mo.eval()
    test_Loss = 0
    predictions = 0
    with torch.no_grad():
        for i, j in dataloader:
            k, l = i.to(device), j.to(device)
            pred = vgg_Mo(k.float())
            test_Loss += loss_Func(pred, l.long()).item()
            predictions += (torch.argmax(pred, dim = 1) == l.to(device)).type(torch.float).sum().item()
    test_Loss /= size 
    test_Accuracy = 100 * (predictions / size)
#     print(f"Test Accuracy: {(test_Accuracy):>0.1f}%, Avg Testing Loss: {test_Loss:>8f} \n")
    
    return test_Loss, test_Accuracy

torch.cuda.empty_cache() 

# This will load the pretrained model from PyTorch library...
vgg16 = models.vgg16_bn(weights = "IMAGENET1K_V1")

# This will change the final layer of VGG16 Model for Transfer Learning...
fc_features = vgg16.classifier[6].in_features

 # This will change the output to loaded classes in the dataset...
vgg16.classifier[6] = nn.Linear(fc_features, tot_Cls)

# This will move the model to available device...
vgg16 = vgg16.to(device)
pre_pruned_accuracy = 0 
post_pruned_accuracy = 0 
pre_pruned_size = sum(p.numel() for p in vgg16.parameters())  
post_pruned_size = 0

epochs = 5
learning_Rate = 1e-6
loss_Func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr = learning_Rate)

total_Train_Loss_VGG16bn = []
total_Test_Loss_VGG16bn = []
total_Val_loss_VGG16bn = []

total_Val_Accuracies_VGG16bn = []
total_Test_Accuracies_VGG16bn = []
total_Train_Accuracies_VGG16bn = []

pre_pruned_accuracies = []
post_pruned_accuracies = []
pre_pruned_sizes = []
post_pruned_sizes = []

best_val_loss = float("inf")
this_is_too_much = 3
too_much = 0

start = time.time()

# The actual training, testing is executed from here, it will run the loop till assigned epochs...
for t in range(epochs):
    
    start1 = time.time()
    print(f"Epoch {t+1}\n-------------------------------")
    tr_L, tr_A = train(train_Dataloader, vgg16, loss_Func, optimizer)
    print(f"Training Accuracy: {(tr_A):>0.1f}%, Avg loss: {tr_L:>8f} \n")
    vl_L, vl_A = validate(val_Dataloader, vgg16, loss_Func)
    print(f"Validation Accuracy: {(vl_A):>0.1f}%, Avg loss: {vl_L:>8f} \n")
    te_L, te_A = test(test_Dataloader, vgg16, loss_Func)
    print(f"Testing Accuracy: {(te_A):>0.1f}%, Avg loss: {te_L:>8f} \n")
          
    total_Train_Loss_VGG16bn.append(tr_L)
    total_Train_Accuracies_VGG16bn.append(tr_A)
    total_Val_loss_VGG16bn.append(vl_L)
    total_Val_Accuracies_VGG16bn.append(vl_A)
    total_Test_Loss_VGG16bn.append(te_L)
    total_Test_Accuracies_VGG16bn.append(te_A)
    
    # This will save the model's state before pruning...
    saved_state = vgg16.state_dict().copy()
    
    # This will prune the model...
    pre_pruned_sizes.append(sum(p.numel() for p in vgg16.parameters()))
    print("Model non-zero parameters before pruning: ", sum(p.nonzero(as_tuple=True)[0].size(0) for p in vgg16.parameters()))
    vgg16 = prune_model(vgg16, amount = 0.6)
    print("Model non-zero parameters post pruning: ", sum(p.nonzero(as_tuple=True)[0].size(0) for p in vgg16.parameters()))
    # This will Re-Evaluate pruned model...
    post_prune_val_loss, post_prune_val_accuracy = validate(val_Dataloader, vgg16, loss_Func)
    print(f"Validation Accuracy post pruning: {(post_prune_val_accuracy):>0.1f}%, Avg loss: {post_prune_val_loss:>8f} \n")        

    pre_pruned_accuracies.append(vl_A)
    post_pruned_accuracies.append(post_prune_val_accuracy)
    post_pruned_sizes.append(sum(p.numel() for p in vgg16.parameters() if p.requires_grad and p.abs().sum() > 0))

    if vl_L < best_val_loss:
        best_val_loss = vl_L
        too_much = 0
    else:
        print(f"Reverting pruning for Epoch {t+1} due to validation accuracy drop.")
        vgg16.load_state_dict(saved_state)
        too_much += 1

    if too_much >= this_is_too_much:
        print("Early stopping for your own good...")
        print(f"From Epoch {t-2}, model started overfitting the data...")
        break
        
    end1 = time.time()
    elapsed_Time = end1 - start1
    minutes = int(elapsed_Time // 60)
    seconds = int(elapsed_Time % 60)
    print(f"Time taken to complete this epoch... {minutes} mins {seconds} secs")
    print()

torch.save(vgg16.state_dict(), "/projects/academic/courses/cse676smr23/deekayG/smallfiles_dontworry/Models_Checkpoints/DLA2Bonus/VGG/VGG16bn_Model_Pruned.pt")

end = time.time()
elapsed_Time = end - start
minutes_End = int(elapsed_Time // 60)
seconds_End = int(elapsed_Time % 60)

print(f"Time taken to complete the whole training... {minutes_End} mins {seconds_End} secs")


# This will plot Model Size Before and After Pruning...
epochs_range = list(range(1, epochs + 1))
plt.plot(epochs_range, pre_pruned_sizes, label='Pre-Pruned', marker='o', color='blue')
plt.plot(epochs_range, post_pruned_sizes, label='Post-Pruned', marker='o', color='green')
plt.xlabel('Epochs')
plt.ylabel('Model Size (Number of Parameters)')
plt.title('Model Size Before and After Pruning')
plt.legend()
plt.savefig("Size.png")

# This will plot Model Accuracy Before and After Pruning...
plt.plot(epochs_range, total_Val_Accuracies_VGG16bn, label='Pre-Pruned', marker='o', color='blue')
plt.plot(epochs_range, post_pruned_accuracies, label='Post-Pruned', marker='o', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.title('Model Accuracy Before and After Pruning')
plt.legend()
plt.savefig("Acc.png")


# This will plot Training, Validation, & Testing Accuracies...
plt.figure(figsize = (7, 7))
plt.plot(total_Train_Accuracies_VGG16bn, label = "Train Accuracy")
plt.plot(total_Val_Accuracies_VGG16bn, label = "Validation Accuracy")
plt.plot(total_Test_Accuracies_VGG16bn, label = "Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metrics")
plt.legend()
plt.savefig("All_Accuracies.png")

# This will plot Training, Validation, & Testing Losses graph...
plt.figure(figsize = (7, 7))
plt.plot(total_Train_Loss_VGG16bn, label = "Train Loss")
plt.plot(total_Val_loss_VGG16bn, label = "Validation Loss")
plt.plot(total_Test_Loss_VGG16bn, label = "Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Metrics")
plt.legend()
plt.savefig("All_Losses.png")

# This will plot Training and Validation Accuracy graph...
plt.figure(figsize = (7, 7))
plt.plot(total_Train_Accuracies_VGG16bn, label = "Train Accuracy")
plt.plot(total_Val_Accuracies_VGG16bn, label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Tr_Vl_Acc.png")

# This will plot Training and Validation Loss graph ...
plt.figure(figsize = (7, 7))
plt.plot(total_Train_Loss_VGG16bn, label = "Train Loss")
plt.plot(total_Val_loss_VGG16bn, label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Tr_Vl_Loss.png")

true_Labels = []
features = []
predicted_Labels = []

# This will predict from the saved model to be visualized for confusion matrix...
with torch.no_grad():
    for i, j in test_Dataloader:
        k, l = i.to(device), j.to(device)
        pred = vgg16(k.float())
        features_Pred = vgg16.features(k.float())
        features.extend(features_Pred.cpu().numpy())
        true_Labels.extend(l.cpu().numpy())
        predicted_Labels.extend(torch.argmax(pred, dim = 1).cpu().numpy())

# This will plot the confusion matrix graph...
fig = plt.figure(figsize=(10, 10))
con_Mat = confusion_matrix(true_Labels, predicted_Labels)
sb.heatmap(con_Mat, annot = True, cmap = "Blues", fmt = "g", 
            xticklabels = class_Names, 
                yticklabels = class_Names)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.savefig("Conf_Mat.png")

# This will plot the F1-Score bars for each class...
fig = plt.figure(figsize = (10, 10))
f1_Class = f1_score(true_Labels, predicted_Labels, average = None)
length = np.arange(len(class_Names))
bars = plt.bar(length, f1_Class, align = "center", alpha = 0.7)
plt.xticks(length, class_Names)
plt.ylabel("F1-score")
plt.title("Class-wise F1-scores")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, 
            f"{height:.2f}", ha = "center", va = "bottom")
plt.savefig("F1-Score.png")
