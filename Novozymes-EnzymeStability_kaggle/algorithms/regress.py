import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from utilities.runUtils import START_SEED


class MLPRegress(nn.Module):
    '''
    Simple MLP network
    '''
    def __init__(self, input_dim, out_dim,
                    layers = 2, dropout = 0,):
        super(MLPRegress, self).__init__()
        START_SEED()

        self.num_conn = np.geomspace(input_dim, out_dim, num=layers+1).astype(int)

        fcu = []
        for i in range(1, len(self.num_conn)):
            fcu.append( nn.ReLU() )
            fcu.append( nn.BatchNorm1d(self.num_conn[i-1]) )
            fcu.append( nn.Linear(self.num_conn[i-1],
                                    self.num_conn[i]))

        self.fc_blocks = nn.Sequential(*fcu)
        self.dropout =nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.dropout(self.fc_blocks(x))
        return out


if __name__ == "__main__":

    model = MLPRegress(1024, 1, layers=3)
    summary(model, (3, 1024), depth=2 )