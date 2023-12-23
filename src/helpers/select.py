
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
import torch
import torch.nn.functional as F

def select(x: Int[Tensor, "batch d"], selection: Int[Tensor, "batch"],):
    """select inds from the 2nd dim of a tensor"""
    B = x.shape[0]
    batch_inds = (torch.arange(B)
                  .long()
                  .unsqueeze(0)
                  .to(selection.device)
                  )
    return x[batch_inds, selection.long()].squeeze(0)

def select_loop(x: Int[Tensor, "batch d"], selection: Int[Tensor, "batch"],):
    """select inds from the 2nd dim of a tensor"""
    batch_size = selection.shape[0]
    y = []
    for i in range(batch_size):
        y.append(x[i, selection[i]])
    return torch.stack(y)



def select2(x: Int[Tensor, "batch d"], selection: Int[Tensor, "batch a"],):
    """select inds from the 2nd dim of a tensor"""
    B = x.shape[0]
    batch_inds = (torch.arange(B)
                  .long()
                  .unsqueeze(1)
                  .to(selection.device)
                  )
    return x[batch_inds, selection.long()]#.squeeze(2)

def select_loop2(x: Int[Tensor, "batch d"], selection: Int[Tensor, "batch a"],):
    y = [select_loop(x, selection[:, i]) for i in range(selection.shape[1])]
    return torch.stack(y).T


if __name__ == '__main__':
    import numpy as np

    # UNIT TEST
    B = 3
    lie_label = torch.randint(2, (B, )).long()
    choice_ids1 = torch.randint(500, (B, 2))
    print(choice_ids1.shape, lie_label.shape)
    print(choice_ids1, lie_label)
    A = select(choice_ids1, lie_label)
    B = select_loop(choice_ids1, lie_label)
    print('A, B', A, B)
    np.testing.assert_array_equal(A, B)


    # UNIT TEST
    B = 3
    lie_label = torch.randint(2, (B, 4)).long()
    choice_ids1 = torch.randint(500, (B, 2))
    print(choice_ids1.shape, lie_label.shape)
    print(choice_ids1, lie_label)
    B = select_loop2(choice_ids1, lie_label)
    A = select2(choice_ids1, lie_label)
    print('A, B', A, B)
    np.testing.assert_array_equal(A, B)
