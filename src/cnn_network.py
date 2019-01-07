import torch.nn as nn
import torch.nn.functional as F

class CNN_Network(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.cv1=nn.Conv2d(3,16,5,padding=2)
        self.cv2=nn.Conv2d(16,32,3,padding=1)
        self.cv3=nn.Conv2d(32,64,3,padding=1)
        self.cv4=nn.Conv2d(64,128,3,padding=1)
        self.cv5=nn.Conv2d(128,256,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(7*7*256,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,133)
        self.dropout=nn.Dropout(0.2)
    
    def forward(self, x):
        x=self.pool(F.relu(self.cv1(x)))
        x=self.dropout(x)
        x=self.pool(F.relu(self.cv2(x)))
        x=self.dropout(x)
        x=self.pool(F.relu(self.cv3(x)))
        x=self.dropout(x)
        x=self.pool(F.relu(self.cv4(x)))
        x=self.dropout(x)
        x=self.pool(F.relu(self.cv5(x)))
        x=self.dropout(x)
        x=x.view(-1,256*7*7)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=F.relu(self.fc2(x))
        x=self.dropout(x)
        x=self.fc3(x)
        return x