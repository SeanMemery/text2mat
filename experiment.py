import torch
from torch import optim
import pytorch_lightning as pl
from torch import tensor as Tensor
import mlflow

from common_rendering import *

class Experiment(pl.LightningModule):

    def __init__(self,
                 model,
                 app,
                 params: dict,
                 opt,
                 data_params) -> None:
        super(Experiment, self).__init__()

        self.model = model
        self.opt = opt
        self.params = params
        self.app = app
        self.lr = self.params['LR']

    def training_epoch_end(self, outputs):
        mlflow.pytorch.log_model(self.model, "model", pip_requirements="./requirements.txt")

    def training_step(self, batch, batch_idx, optimizer_idx = 0):  
        batch = self.model.gen_random_batch(self.app, False)
        results = self.model.forward(batch[0])

        train_loss = self.model.loss_function(*results,
                                              batch = batch,
                                              app = self.app,
                                              loss_config = self.params,
                                              epoch = self.current_epoch,
                                              validation = False)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        batch = self.model.gen_random_batch(self.app, True)
        results = self.model.forward(batch[0])

        val_loss = self.model.loss_function(*results,
                                            app = self.app,
                                            batch = batch,
                                            loss_config = self.params,
                                            epoch = self.current_epoch,
                                            validation = True)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):

        if self.opt == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.params['LR'])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "Adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "Adadelta":
            optimizer = optim.Adadelta(self.model.parameters(),
                                       lr=self.lr,
                                       weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "Adagrad":
            optimizer = optim.Adagrad(self.model.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "AdamW":
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "Adamax":
            optimizer = optim.Adamax(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "ASGD":
            optimizer = optim.ASGD(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "RMSprop":
            optimizer = optim.RMSprop(self.model.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "Rprop":
            optimizer = optim.Rprop(self.model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        elif self.opt == "LBFGS":
            optimizer = optim.LBFGS(self.model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.params['weight_decay'])

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return [optimizer], [scheduler]
        else:
            print(f"Opt: {self.opt}")
            raise ValueError("Unknown optimizer")
