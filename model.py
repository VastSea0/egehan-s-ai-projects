import torch 
import torch.nn as nn

class MyNerualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # takes inputs, outputs to first hidden layer
        self.layer1 = nn.Linear(input_size, hidden_size)

        # takes first hidden layer outputs, outputs to second hidden layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        self.layer3 = nn.Linear(hidden_size, hidden_size)

        # takes second hidden layer outputs, outputs final prediction
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x= self.output_layer(x)
        return(x)

model = MyNerualNetwork(input_size = 10, hidden_size = 16, output_size=1)

dummy_input = torch.randn(1,10)

prediction = model(dummy_input)

print("output tensor shape: ", prediction.shape)