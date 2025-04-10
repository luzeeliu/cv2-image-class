import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from CNN_model import CONV_ResNet
from CNN_image_process import get_dataloader, transform

def train_one_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()
    total_loss, correct = 0, 0
    for image, label in dataloader:
        # take image and label form dataloader and put them to GPU
        image = image.to(device)
        label = label.to(device)
        
        output = model(image)
        loss = loss_func(output, label)
        
        # set optimal
        # change gradient during training time
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * image.size(0)
        # correct will sum all dimention = 1
        correct += (output.argmax(1) == label).sum().item()
    
    #return the acc and loss
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


# validation function to test overfitting 
def validation(model, dataloader, loss_func, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
            loss = loss_func(output, label)
            
            total_loss += loss.item() * image.size(0)
            correct += (output.argmax(1) == label).sum().item()
            
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

if __name__ == "__main__":
    # base value
    trainData_path = os.path.join("dataset", "train")
    testData_path = os.path.join("dataset", "test")
    save_path = os.path.join("net", "CNN_RESULT.pth")
    batch_size = 10
    image_size = 224
    epochs = 35
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data process
    tran_transform, test_transform = transform(input_size= 224)
    train_data, classes = get_dataloader(trainData_path, transform= tran_transform)
    test_data, classes = get_dataloader(testData_path, transform= test_transform)
    
    # dataloader
    train_loader = DataLoader(train_data, batch_size = len(classes), shuffle = True)
    val_loader = DataLoader(test_data, batch_size = len(classes), shuffle = False)
    model = CONV_ResNet(num_classes= len(classes)).to(device)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    #set epoch
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,optimizer, loss_func, device)
        val_loss, val_acc = validation(model,val_loader, loss_func, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
      
    torch.save(model.state_dict(), save_path)
    print("model saved")
        
        