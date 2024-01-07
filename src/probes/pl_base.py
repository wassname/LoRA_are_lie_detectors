from pytorch_optimizer import Ranger21
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch import optim

class PLBase(pl.LightningModule):
    """
    Base pytorch lightning module, subclass to add model
    """
    def __init__(self, steps_per_epoch: int, max_epochs: int, lr=4e-3, weight_decay=1e-9):
        super().__init__()
        self.probe = None # subclasses must add this
        self.total_steps = steps_per_epoch * max_epochs
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
        
    def _step(self, batch, batch_idx, stage='train'):
        x0, y = batch

        ypred0 = self(x0)
        
        if stage=='pred':
            return ypred0
        
        loss = F.smooth_l1_loss(ypred0, y)
        
        y_cls = ypred0 > 0.5
        self.log(f"{stage}/acc", accuracy(y_cls, y>0.5, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss
    
    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx=0):
        return self._step(batch, batch_idx, stage='val')
    
    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage='pred').cpu().detach()
    
    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage='test')
    
    # def configure_optimizers(self):
    #     """use ranger21 from  https://github.com/kozistr/pytorch_optimizer"""
    #     optimizer = Ranger21(
    #         self.parameters(),
    #         lr=self.hparams.lr,
    #         weight_decay=self.hparams.weight_decay,       
    #         num_iterations=self.hparams.steps_per_epoch * self.hparams.max_epochs,
    #     )
    #     return optimizer
    
    def configure_optimizers(self):
        """simple vanilla torch optim"""
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html#quantization-via-bitsandbytes
        # optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        total_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs
        # if self.ae_mode = 0
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, self.hparams.lr, total_steps=total_steps, verbose=False
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
