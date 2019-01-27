
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
import json
from collections import OrderedDict
from PIL import Image
import numpy as np


arch = {"vgg16":25088,"densenet121" :1024}


# In[ ]:


def load_data(path_part): 
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validationloader=torch.utils.data.DataLoader(validation_data, batch_size=32)
    return trainloader, validationloader, testloader


# In[ ]:


def nn_architecture(architecture, dropout , fc2, learn_r, gpu_cpu):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True) 
    else:
        print('please choose either vgg16 or densenet121')
        
    for param in model.parameters():
        param.requires_grad = False
        num_inputs = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                         ('fc1',nn.Linear(25088,1024)),
                         ('relu1',nn.ReLU()),
                         ('dropout1',nn.Dropout(0.5)),
                         ('fc2',nn.Linear(1024,102)),
                         ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        if gpu_cpu == 'gpu':
            model.to(device = 'cuda')
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = learn_r)

    return model, optimizer, criterion


# In[ ]:


def train_network(model, criterion, optimizer, loader, validationloader, epoch, val_step, gpu_cpu):
    model.to('cuda')
    for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if steps % val_step == 0:
            model.eval()
            validation_loss = 0
            validation_accuracy = 0
            
            for ii, (inputs2,labels2) in enumerate(validationloader):
                optimizer.zero_grad()
                
                inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                model.to('cuda:0')
                with torch.no_grad():    
                    outputs = model.forward(inputs2)
                    validation_loss = criterion(outputs,labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    validation_accuracy += equality.type_as(torch.FloatTensor()).mean()
            
            validation_loss = validation_loss / len(validationloader)
            validation_accuracy = validation_accuracy / len(validationloader)
       
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/val_step),
                   "Validation Loss {:.4f}".format(validation_loss),
                   "Validation Accuracy: {:.2f}".format(validation_accuracy))


# In[ ]:


def save_checkpoint(filepath , architecture, dropout, learn_r, fc2, epochs):
    model.class_to_idx = train_data.class_to_idx
    #epochs=5
    checkpoint = ({'Pretrained': models.vgg16(pretrained=True),
               'state_dict': model.state_dict(), 
               'class_to_idx': model.class_to_idx,
               'optimizer_state_dict': optimizer.state_dict(),
               'epochs': epochs, 
               'classifier': model.classifier})
    torch.save(checkpoint, 'checkpoint.pth')
    print ('model saved')


# In[ ]:


def load_checkpoint_rebuild_model(filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model = checkpoint['Pretrained'] 
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    return model


# In[ ]:


def process_image(image):
    img = Image.open(image)
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    processed_img = transformations(img)
    return(processed_img)


# In[ ]:


def predict(image_path, model, topk, gpu_cpu):
    if torch.cuda.is_available() and gpu_cpu == 'gpu':
        model.to('cuda:0')

    img = process_image(image_path)
    img = img.unsqueeze(0)
    img = img.float()
    if gpu_cpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    probability = F.softmax(output.data, dim = 1)
    return probability.topk(topk)

