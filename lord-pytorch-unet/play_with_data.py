import torch

if __name__ == '__main__':
    a = torch.tensor([[[6,6],[2,2]]])
    b = torch.tensor([[[3,3],[1/2,1/2]]])
    c = a*b
    print(c)