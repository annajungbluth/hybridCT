from emulator.mimo.pyct.Model import UNet
import torch
from torch import nn
import lightning.pytorch as pl
from typing import Dict, Any

def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CloudUNetPL(pl.LightningModule):

    def __init__(self, config, in_channels=9, out_channels=9, init_features=64):
        self.config = config
        super().__init__()

        # set up model
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)

        self.save_hyperparameters()
        self.save_hyperparameters({
            "trainable_params": count_trainable_parameters(self.model),
        })


    def loss(self, y_hat, y):
        return torch.sqrt(torch.nn.functional.mse_loss(y_hat, y))

    def compile(self):
        """Compile the model."""
        self.model = torch.compile(self.model)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        image, label = batch["image"], batch["label"]
        y_hat = self.model(image)
 
        loss = self.loss(y_hat, label)
        self.log("train_loss", loss.mean(), batch_size=self.trainer.datamodule.batch_size)

        return {"loss": loss}

    def forward(self, x):
        y_hat = self.model(x)

        return y_hat


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        image, label = batch["image"], batch["label"]
        y_hat = self.model(image)

        val_loss = self.loss(y_hat, label)
        self.log("val_loss", val_loss.mean(), batch_size=self.trainer.datamodule.batch_size)
        return {"val_loss": val_loss}


    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["scheduler_step_size"],
            gamma=self.config["scheduler_gamma"],
            verbose=True,
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            monitor="val_loss",
        )




