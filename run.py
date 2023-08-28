import os, yaml, argparse, sys, shutil

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import mlflow
from data.VDB import VDB

from experiment import Experiment
from model import ModularModel

from common_rendering import *

class RandGeneralDataset(Dataset):
    def __init__(self, split="train"):
        self.size = 1024 if split == "train" else 8

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return -1
    
class ModularDataset(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 8
        self.pin_memory = True
        self.shuffle = True

        self.train_dataset = RandGeneralDataset(
            split="train",
            **kwargs,
        )

        self.val_dataset = RandGeneralDataset(
            split="test",
            **kwargs,
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
        ) 

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        ) 

def run_experiment(config, args, app=None):

    mlflow_uri = "./logs/mlflow"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(config['model_params']['name'])
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=config['logging_params']['name']):

        params = dict( **config["model_params"], **config["data_params"], **config["exp_params"] )
        mlflow.log_params(params)
        
        print(f"MLFlow run ID: {mlflow.active_run().info.run_id}")

        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
        )
        
        seed_everything(config['exp_params']['manual_seed'], True)
        if args.id is None:
            model = ModularModel(**config['model_params'], data_params=config['data_params'], exp_params=config['exp_params'])
        else:
            model_path = f"./logs/mlflow/0/{args.id}/artifacts/model"
            model = mlflow.pytorch.load_model(model_path)
            model.training_images = config["model_params"]["training_images"]

        clip_version = config['model_params']['clip_version']
        model.vdb = VDB(clip_version)

        if app is None:
            app = create_app(config)

        experiment = Experiment(model, app, config['exp_params'], config["model_params"]["opt"], config['data_params'])
        data = ModularDataset(**config["data_params"])

        ### Make validation directories
        val_dir = config['logging_params']['val_dir']
        shutil.rmtree(val_dir) if os.path.exists(val_dir) else None
        os.makedirs(val_dir)
           
        callbacks=[]

        # model_path = f"./logs/mlflow/{mlf_logger.experiment_id}/{mlf_logger.run_id}/checkpoints/"
        # callbacks.append(ModelCheckpoint(dirpath = model_path, 
        #                     monitor="val_loss",
        #                     save_top_k=1,))

        ### Learning Rate Monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))

        ### Progress Bar
        callbacks.append(RichProgressBar(
                    theme=RichProgressBarTheme(
                    description="green_yellow",
                    progress_bar="green1",
                    progress_bar_finished="green1",
                    progress_bar_pulse="#6206E0",
                    batch_progress="green_yellow",
                    time="grey82",
                    processing_speed="grey82",
                    metrics="grey82",
                )
            )
        )

        if args.loop:
            try:
                runner = Trainer(callbacks=callbacks, logger=mlf_logger, **config['trainer_params'])
                print(f"======= Training {config['model_params']['name']} =======")
                print(f"MLFlow run ID: {mlflow.active_run().info.run_id}")
                runner.fit(experiment, datamodule=data)
                return runner, experiment
            except Exception as e:
                print(e)
                while True:
                    try:
                        print("Training Error, retrying ...")
                        try:
                            model_path = f"./logs/mlflow/0/{args.id}/artifacts/model"
                            model = mlflow.pytorch.load_model(model_path)
                        except:
                            model = ModularModel(**config['model_params'], data_params=config['data_params'], exp_params=config['exp_params'])
                        experiment = Experiment(model, app, config['exp_params'], config["model_params"]["opt"], config['data_params'])
                        runner = Trainer(callbacks=callbacks, logger=mlf_logger, **config['trainer_params'])
                        print(f"======= Training {config['model_params']['name']} =======")
                        print(f"MLFlow run ID: {mlflow.active_run().info.run_id}")
                        runner.fit(experiment, datamodule=data)
                        return runner, experiment
                    except Exception as e:
                        print(e)
                        continue
        else:
            runner = Trainer(callbacks=callbacks, logger=mlf_logger, **config['trainer_params'])
            print(f"======= Training {config['model_params']['name']} =======")
            print(f"MLFlow run ID: {mlflow.active_run().info.run_id}")
            runner.fit(experiment, datamodule=data)
            return runner, experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="config",
                        metavar='FILE',
                        help =  "config file",
                        default=None)
    parser.add_argument('--id', 
                        dest="id",
                        metavar='FILE',
                        help =  "train id",
                        default=None)
    parser.add_argument('--loop', 
                        action='store_true',
                        dest="loop",
                        help =  "failure loop for training",
                        default=False)
    args = parser.parse_args()

    config_path = f"configs/{args.config}.yaml"
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    run_experiment(config, args)
