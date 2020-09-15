import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

def saveModel(model_state, optimizer_state, save_path):
    state = {'state_dict': model_state,
                    'optimizer': optimizer_state }
    torch.save(state, save_path)



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(myLSTM_Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) 
        self.out = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        if use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret
    
    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden