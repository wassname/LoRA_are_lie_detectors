"""
see https://github.com/cornellius-gp/gpytorch/blob/master/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.ipynb
"""


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        # variational_strategy = UnwhitenedVariationalStrategy(
        #     self, train_x, variational_distribution, learn_inducing_locations=False
        # )
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(train_x.size(0))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

class PLGP(PLRankingBase):
    def __init__(self, epoch_steps: int, max_epochs: int, lr=4e-3, weight_decay=1e-9):
        super().__init__(epoch_steps=epoch_steps, max_epochs=max_epochs, lr=lr, weight_decay=weight_decay)

        # Initialize model and likelihood
        self.model = GPClassificationModel(X_train)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()


        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the number of training datapoints
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model,num_data=epoch_steps)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, stage="train"):
        x0, y = batch

        ydist = self.model(x0)

        ypred = self.likelihood(self(x0)).probs
        
        if stage=='pred':
            return ypred
        
        loss = -self.mll(ydist, y)
        
        self.log(f"{stage}/acc", accuracy(ypred, y, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss

def run():
    dl_train = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    max_epochs = 50

    pl_gp = PLGP(epoch_steps=len(X_train), max_epochs=max_epochs, lr=4e-3, weight_decay=0)
    trainer = pl.Trainer(
        gradient_clip_val=20,
        accelerator="auto",
        # devices="1",
        max_epochs=max_epochs,
        log_every_n_steps=1,
        # enable_progress_bar=verbose, enable_model_summary=verbose
    )

    trainer.fit(model=pl_gp, train_dataloaders=dl_train, val_dataloaders=dl_val);
    df_hist, _ = read_metrics_csv(trainer.logger.experiment.metrics_file_path)
    df_hist[['val/loss_epoch', 'train/loss_epoch']].plot()
    df_hist[['val/acc', 'train/acc']].plot()

