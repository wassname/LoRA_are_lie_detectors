import torch
import torch.nn as nn

noop = lambda x: x

class LinBnDrop(nn.Module):
    "A linear layer with `nn.BatchNorm1d` and `nn.Dropout`."
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, **kwargs):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out) if bn else noop
        self.act = nn.ReLU() if act is None else act()
        self.do = nn.Dropout(p) if p else noop

    def forward(self, x):
        x = self.bn(self.act(self.lin(x)))
        return self.do(x)


class HeadBlock(nn.Module):
    def __init__(self, c_in, c_out=1, layers=[], dropout=0):
        super().__init__()
        self.head = []
        layer_sizes = [c_in]+layers+[c_out]
        for i in range(len(layer_sizes)-1):
            a,b = layer_sizes[i], layer_sizes[i+1]
            if i==len(layer_sizes)-1: # last layer
                self.head.append(nn.Linear(a, b))
            else:
                self.head.append(LinBnDrop(a, b, p=dropout))
        self.head = nn.Sequential(*self.head)

    def forward(self, x):
        return self.head(x)



