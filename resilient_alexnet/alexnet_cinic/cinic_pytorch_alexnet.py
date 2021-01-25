from resilient_alexnet.alexnet_fashion.fashion_pytorch_alexnet import Fashion_PyTorch_AlexNet
import pytorch_lightning as pl
import torch

def cinic_pt_objective(config):
    torch.manual_seed(0)
    model = Fashion_PyTorch_AlexNet(config)
    ### edit the data
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0], num_sanity_val_steps=0)
    trainer.fit(model)
    trainer.test(model)
    return (model.test_accuracy, model.model, model.calculated_training_loss, model.calculated_validation_loss,
            model.calculated_validation_acc)