from resilient_alexnet.alexnet_fashion.fashion_pytorch_alexnet import Fashion_PyTorch_AlexNet
import pytorch_lightning as pl
import torch
import pickle
import argparse
from torch import nn

def cinic_pt_objective(config):
    torch.manual_seed(0)
    model = Fashion_PyTorch_AlexNet(config)
    classes = 10
    model.model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
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
    ### edit the data
    ### load in pickled dataset
    f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/cinic_splits.pkl', 'rb')
    data = pickle.load(f)
    (model.x_train, model.y_train), (model.x_val, model.y_val), (model.x_test, model.y_test) = data
    ### perform fitting and testing
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0], num_sanity_val_steps=0)
    trainer.fit(model)
    trainer.test(model)
    return (model.test_accuracy, model.model, model.calculated_training_loss, model.calculated_validation_loss,
            model.calculated_validation_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch")
    args = parser.parse_args()
    if args.batch:
        batch = args.batch
    else:
        batch = 64
    test_config = {'batch_size': batch, 'learning_rate': .0001, 'epochs': 5, 'dropout': 0.5, 'adam_epsilon': 10**-9}
    res = cinic_pt_objective(test_config)
