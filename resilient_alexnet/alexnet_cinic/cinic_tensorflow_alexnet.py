from resilient_alexnet.alexnet_fashion.fashion_tensorflow_alexnet import Fashion_TensorFlow_AlexNet

def cinic_tf_objective(config):
    model = Fashion_TensorFlow_AlexNet(config)
    ### edit the data here
    model.fit()
    accuracy = model.test()
    return accuracy, model.model, model.training_loss_history, model.val_loss_history, model.val_acc_history