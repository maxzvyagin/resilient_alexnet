import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import statistics
import os
import argparse
from hyper_resilient_experiments.alexnet_caltech.caltech_tensorflow_alexnet import get_caltech
import pickle
from torch.utils.data import Dataset
import numpy as np

class Caltech_NP_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        selected_x = torch.from_numpy(self.x[index]).float()
        selected_y = self.y[index].item()
        return selected_x, selected_y

    def __len__(self):
        return len(self.x)

class Caltech_PyTorch_AlexNet(pl.LightningModule):
    def __init__(self, config, classes=102):
        super(Caltech_PyTorch_AlexNet, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            ### 300 x 200
            #nn.Linear(17920, 4096),
            ### 256 x 256
            nn.Linear(16384, 4096),
            #nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(4096, classes))
        self.criterion = nn.CrossEntropyLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()
        ### load in pickled dataset
        f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/caltech_splits.pkl', 'rb')
        data = pickle.load(f)
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = data
        # tracking metrics
        self.training_loss_history = []
        self.validation_loss_history = []
        self.validation_acc_history = []

    def train_dataloader(self):
        return DataLoader(Caltech_NP_Dataset(self.x_train, self.y_train),
                          batch_size=int(self.config['batch_size']), shuffle=False)

    def val_dataloader(self):
        return DataLoader(Caltech_NP_Dataset(self.x_val, self.y_val),
                          batch_size=int(self.config['batch_size']), shuffle=False)

    def test_dataloader(self):
        return DataLoader(Caltech_NP_Dataset(self.x_test, self.y_test),
                          batch_size=int(self.config['batch_size']), shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'],
                                     eps=self.config['adam_epsilon'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return {'forward': self.forward(x), 'expected': y}

    def training_step_end(self, outputs):
        # only use when  on dp
        loss = self.criterion(outputs['forward'], outputs['expected'])
        logs = {'train_loss': loss}
        self.training_loss_history.append(loss.item())
        return {'loss': loss, 'logs': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        return {'forward': self.forward(x), 'expected': y}

    def validation_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'])
        accuracy = self.accuracy(outputs['forward'], outputs['expected'])
        logs = {'validation_loss': loss, 'validation_accuracy': accuracy}
        self.validation_loss_history.append(loss.item())
        self.validation_acc_history.append(accuracy.item())
        return {'validation_loss': loss, 'logs': logs, 'validation_accuracy': accuracy}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'])
        accuracy = self.accuracy(outputs['forward'], outputs['expected'])
        logs = {'test_loss': loss, 'test_accuracy': accuracy}
        return {'test_loss': loss, 'logs': logs, 'test_accuracy': accuracy}

    def test_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['test_loss']))
        avg_loss = statistics.mean(loss)
        tensorboard_logs = {'test_loss': avg_loss}
        self.test_loss = avg_loss
        accuracy = []
        for x in outputs:
            accuracy.append(float(x['test_accuracy']))
        avg_accuracy = statistics.mean(accuracy)
        self.test_accuracy = avg_accuracy
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy}


def caltech_pt_objective(config):
    torch.manual_seed(0)
    model = Caltech_PyTorch_AlexNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0], num_sanity_val_steps=0)
    trainer.fit(model)
    trainer.test(model)
    return (model.test_accuracy, model.model, model.training_loss_history,
            model.validation_loss_history, model.validation_acc_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch")
    args = parser.parse_args()
    if args.batch:
        batch = args.batch
    else:
        batch = 64
    test_config = {'batch_size': batch, 'learning_rate': .0001, 'epochs': 100, 'dropout': 0.5, 'adam_epsilon': 10**-9}
    res = caltech_pt_objective(test_config)
