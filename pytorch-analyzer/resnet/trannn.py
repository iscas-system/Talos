import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile
import resnet50
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ImageFile.LOAD_TRUNCATED_IMAGES=True

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

img_transforms = transforms.Compose([
    transforms.Resize((64,64)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

train_data_path = "/root/github/beginners-pytorch-deep-learning/chapter2/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)


val_data_path = "/root/github/beginners-pytorch-deep-learning/chapter2/val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms, is_valid_file=check_image)

test_data_path = "/root/github/beginners-pytorch-deep-learning/chapter2/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image)

batch_size=64



train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size) 
test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

simplenet = resnet50.resnet50
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda") 
else:
    print("cpu")
    device = torch.device("cpu")
simplenet.to(device)

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
        # print(prof)
        prof.export_chrome_trace("/tmp/resnet50_CPU.json")
        print("Exported trace")
        training_loss /= len(train_loader.dataset)
        model.eval()

        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

train(simplenet, optimizer,torch.nn.CrossEntropyLoss(), train_data_loader,val_data_loader, epochs=1, device=device)

# labels = ['cat','fish']

# img = Image.open("./val/fish/100_1422.JPG") 
# img = img_transforms(img).to(device)


# prediction = F.softmax(simplenet(img), dim=1)
# prediction = prediction.argmax()
# print(labels[prediction])

# torch.save(simplenet, "/tmp/simplenet") 
# simplenet = torch.load("/tmp/simplenet")

# torch.save(simplenet.state_dict(), "/tmp/simplenet")    
# simplenet = SimpleNet()
# simplenet_state_dict = torch.load("/tmp/simplenet")
# simplenet.load_state_dict(simplenet_state_dict)