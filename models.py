## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.pool = nn.MaxPool2d(2)
        
        self.dense1 = nn.Linear(12*12*256, 1000)
        self.dense2 = nn.Linear(1000, 136)
        
        
        
        
        # NAIMISH poper archirecture:
        '''self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.pool = nn.MaxPool2d(2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        self.dense1 = nn.Linear(256*26*26, 1000)
        self.dense2 = nn.Linear(1000, 1000)
        self.dense3 = nn.Linear(1000, 136)
                
        I.uniform_(self.conv1.weight.data)
        I.uniform_(self.conv2.weight.data)
        I.uniform_(self.conv3.weight.data)
        I.uniform_(self.conv4.weight.data)
        I.xavier_normal_(self.dense1.weight.data)
        I.xavier_normal_(self.dense2.weight.data)
        I.xavier_normal_(self.dense3.weight.data)
        '''
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        
        '''
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        #x = self.pool(F.elu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        #x = F.relu(self.dense2(x)))     #I tried Relu/leaky_relu/tanh, but sigmoid gave the least loss
        x = self.dense3(x)
        '''
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
