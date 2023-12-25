"""
https://github.com/lingo-mit/lm-truthfulness/blob/master/lm_truthfulness_gpt-j_sparse.ipynb
"""
class Probe(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.layer = nn.Linear(model_dim, 2)
    
    def forward(self, reprs):
        return self.layer(reprs)
    
    def predict(self, good_reprs, bad_reprs):
        return (self(good_reprs)[:, 1] > self(bad_reprs)[:, 1]).float()
    
    def l1(self):
        return torch.norm(self.layer.weight, 1)
    
@torch.no_grad()
def eval_probe(probe, data_processed):
    return probe.predict(
        data_processed["good_repr"],
        data_processed["bad_repr"]
    ).mean().item()

def train_probe(train_data_processed, val_data_processed, l1=0):
    n_train, model_dim = train_data_processed["good_repr"].shape
    probe = Probe(model_dim).float().cuda()
    objective = nn.CrossEntropyLoss()
    opt = optim.Adam(probe.parameters(), lr=0.003)
    for i in range(200):
        loss_good = objective(probe(train_data_processed["good_repr"][:-N_DEV].float()), torch.ones(n_train-N_DEV).long().cuda())
        loss_bad = objective(probe(train_data_processed["bad_repr"][:-N_DEV].float()), torch.zeros(n_train-N_DEV).long().cuda())
        loss = loss_good + loss_bad + l1 * probe.l1()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if False:
            if (i+1) % 10  == 0:
                print(loss.item())
    return probe.half()
