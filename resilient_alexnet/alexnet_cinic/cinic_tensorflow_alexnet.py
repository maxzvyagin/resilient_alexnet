from resilient_alexnet.alexnet_fashion.fashion_tensorflow_alexnet import Fashion_TensorFlow_AlexNet
import pickle
import tensorflow as tf

def cinic_tf_objective(config):
    model = Fashion_TensorFlow_AlexNet(config)
    ### edit the data here
    f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/cinic_splits.pkl', 'rb')
    data = pickle.load(f)
    (model.x_train, model.y_train), (model.x_val, model.y_val), (model.x_test, model.y_test) = data
    f.close()
    model.train_data = tf.data.Dataset.from_tensor_slices((model.x_train, model.y_train)).batch(int(config['batch_size']))
    model.val_data = tf.data.Dataset.from_tensor_slices((model.x_val, model.y_val)).batch(int(config['batch_size']))
    model.test_data = tf.data.Dataset.from_tensor_slices((model.x_test, model.y_test)).batch(int(config['batch_size']))
    ### perform fitting and testing
    model.fit()
    accuracy = model.test()
    return accuracy, model.model, model.training_loss_history, model.val_loss_history, model.val_acc_history