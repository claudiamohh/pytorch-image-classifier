    ''' Linear CNN Model '''
class LinearCNN():
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 120)      #input has to be image size to fit 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
