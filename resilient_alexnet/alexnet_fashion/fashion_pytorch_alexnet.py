import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import statistics
import os
import argparse
import pickle

from torch.utils.data import Dataset
import torch
import numpy as np

class Fashion_NP_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        selected_x = torch.from_numpy(self.x[index]).float()
        selected_y = self.y[index].item()
        return selected_x, selected_y

    def __len__(self):
        return len(self.x)


class Fashion_PyTorch_AlexNet(pl.LightningModule):
    def __init__(self, config, classes=10):
        super(Fashion_PyTorch_AlexNet, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(256, 4096),
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
        f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/fashion_splits.pkl', 'rb')
        data = pickle.load(f)
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = data
        # tracking metrics
        self.training_loss_history = []
        self.calculated_training_loss = []
        self.validation_loss_history = []
        self.calculated_validation_loss = []
        self.validation_acc_history = []
        self.calculated_validation_acc = []

    def train_dataloader(self):
        return DataLoader(Fashion_NP_Dataset(self.x_train.astype(np.float32), self.y_train.astype(np.float32)),
                          batch_size=int(self.config['batch_size']), shuffle=False)

    def val_dataloader(self):
        return DataLoader(Fashion_NP_Dataset(self.x_val.astype(np.float32), self.y_val.astype(np.float32)),
                          batch_size=int(self.config['batch_size']), shuffle=False)

    def test_dataloader(self):
        return DataLoader(Fashion_NP_Dataset(self.x_test.astype(np.float32), self.y_test.astype(np.float32)),
                          batch_size=int(self.config['batch_size']), shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_epsilon'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return {'forward': self.forward(x), 'expected': y.long()}

    def training_step_end(self, outputs):
        # only use when  on dp
        loss = self.criterion(outputs['forward'], outputs['expected'])
        logs = {'train_loss': loss}
        # print(type(loss.item()))
        self.training_loss_history.append(loss.item())
        return {'loss': loss, 'logs': logs}

    def training_epoch_end(self, outputs):
        ave = statistics.mean(self.training_loss_history)
        self.training_loss_history = []
        self.calculated_training_loss.append(ave)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        return {'forward': self.forward(x), 'expected': y.long()}

    def validation_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'])
        accuracy = self.accuracy(outputs['forward'], outputs['expected'])
        logs = {'validation_loss': loss, 'validation_accuracy': accuracy}
        # print(type(loss.item()))
        # print(type(accuracy.item()))
        self.validation_loss_history.append(loss.item())
        self.validation_acc_history.append(accuracy.item())
        return {'validation_loss': loss, 'logs': logs, 'validation_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        try:
            # average validation loss
            ave = statistics.mean(self.validation_loss_history)
            self.calculated_validation_loss.append(ave)
            self.validation_loss_history = []
            # average validation accuracy
            ave = statistics.mean(self.validation_acc_history)
            self.calculated_validation_acc.append(ave)
            self.validation_acc_history = []
        except:
            print("Was not able to calculate validation metrics, continuing.")

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y.long()}

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


def fashion_pt_objective(config):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.manual_seed(0)
    model = Fashion_PyTorch_AlexNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, num_sanity_val_steps=0)
    trainer.fit(model)
    trainer.test(model)
    # print(len(model.calculated_training_loss), len(model.calculated_validation_loss),
    #       len(model.calculated_validation_acc))
    return (model.test_accuracy, model.model, model.calculated_training_loss, model.calculated_validation_loss,
            model.calculated_validation_acc)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-b", "--batch")
    # args = parser.parse_args()
    # if args.batch:
    #     batch = args.batch
    # else:
    #     batch = 64
    test_config = {'batch_size': 64, 'learning_rate': .0001, 'epochs': 5, 'dropout': 0.5, 'adam_epsilon': 10**-9}
    res = fashion_pt_objective(test_config)
