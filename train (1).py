import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import arg_train

args = arg_train.get_args()
print(args)

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Build the pre-trained model
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif args.arch == 'vgg19_bn':
    model = models.vgg19_bn(pretrained=True)

# Froze the grad
for param in model.parameters():
    param.requires_grad = False
    
# build the classifier
classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier

def validation(model, validloader, criterion):
    if args.gpu:
        model.to('cuda')
    valid_loss = 0
    accuracy = 0
    for data in validloader:
        images, labels = data
        if args.gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


print_every = 30
steps = 0

if args.gpu:
    model.to('cuda')

for e in range(args.epochs):
    model.train()
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1

        if args.gpu:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()

model.eval()
    
with torch.no_grad():
    _, accuracy = validation(model, testloader, criterion)
                
print("Test Accuracy: {:.2f}%".format(accuracy*100/len(testloader)))

model.class_to_idx = train_data.class_to_idx

torch.save({
            'class_to_idx': model.class_to_idx,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'hidden_units': args.hidden_units,
            'optim_state': optimizer.state_dict()
        }, args.save_dir)