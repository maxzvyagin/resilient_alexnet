import sys
from resilient_alexnet.alexnet_fashion import fashion_pytorch_alexnet, fashion_tensorflow_alexnet
from resilient_alexnet.alexnet_caltech import caltech_pytorch_alexnet, caltech_tensorflow_alexnet
from resilient_alexnet.alexnet_cinic import cinic_pytorch_alexnet, cinic_tensorflow_alexnet
from resilient_alexnet.alexnet_caltech.caltech_pytorch_alexnet import Caltech_NP_Dataset
from resilient_alexnet.alexnet_fashion.fashion_pytorch_alexnet import Fashion_NP_Dataset
import argparse
import ray
from ray import tune
import statistics
import foolbox as fb
import tensorflow as tf
import torch
import torchvision
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import spaceray
from ray.tune.integration.wandb import wandb_mixin
import wandb
from ray.tune.integration.wandb import wandb_mixin
import pickle
import os

# Default constants
PT_MODEL = fashion_pytorch_alexnet.fashion_pt_objective
TF_MODEL = fashion_tensorflow_alexnet.fashion_tf_objective
MODEL_TYPE = "fashion"
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True
MAX_DIFF = False
FASHION = False
MIN_RESILIENCY = False
ONLY_CPU = False
OPTIMIZE_MODE = "max"
MAXIMIZE_CONVERGENCE = False
MODEL_FRAMEWORK = "pt"

def found_convergence(validation_accuracy):
    """Given validation accuracy, return bool defining if convergence has been reached: <=5% change in last 10 points"""
    last_ten = validation_accuracy[-10:]
    diffs = []
    for x in range(9):
        d = last_ten[x+1] - last_ten[x]
        diffs.append(abs(d))
    ave_diff = statistics.mean(diffs)
    if ave_diff >= .05:
        return False, ave_diff
    else:
        return True, ave_diff

def model_attack(model, model_type, attack_type, config, num_classes=NUM_CLASSES):
    print(num_classes)
    global ONLY_CPU, MODEL_TYPE
    if model_type == "pt":
        if ONLY_CPU:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
        # fashion
        if MODEL_TYPE == "fashion":
            f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/fashion_splits.pkl', 'rb')
            data = pickle.load(f)
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
            data = DataLoader(Fashion_NP_Dataset(x_test.astype(np.float32), y_test.astype(np.float32)),
                          batch_size=int(config['batch_size']), shuffle=False)
        elif MODEL_TYPE == "caltech":
            f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/caltech_splits.pkl', 'rb')
            data = pickle.load(f)
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
            data = DataLoader(Caltech_NP_Dataset(x_test.astype(np.float32), y_test.astype(np.float32)),
                              batch_size=int(config['batch_size']), shuffle=False)
        else:
            f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/cinic_splits.pkl', 'rb')
            data = pickle.load(f)
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
            data = DataLoader(Fashion_NP_Dataset(x_test.astype(np.float32), y_test.astype(np.float32)),
                              batch_size=int(config['batch_size']), shuffle=False)
        images, labels = [], []
        for sample in data:
            images.append(sample[0].to(device))
            labels.append(sample[1].to(device))
        # images, labels = (torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device))
    elif model_type == "tf":
        fmodel = fb.models.TensorFlowModel(model, bounds=(0, 1))
        if MODEL_TYPE == "fashion":
            f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/fashion_splits.pkl', 'rb')
            data = pickle.load(f)
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
            data = tf.data.Dataset.from_tensor_slices((x_test.astype(np.float32), y_test.astype(np.float32))).batch(config['batch_size'])
        elif MODEL_TYPE == "caltech":
            f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/caltech_splits.pkl', 'rb')
            data = pickle.load(f)
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
            data = tf.data.Dataset.from_tensor_slices((x_test.astype(np.float32), y_test.astype(np.float32))).batch(config['batch_size'])
        else:
            f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/cinic_splits.pkl', 'rb')
            data = pickle.load(f)
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
            data = tf.data.Dataset.from_tensor_slices((x_test.astype(np.float32), y_test.astype(np.float32))).batch(
                config['batch_size'])
        images, labels = [], []
        for sample in data:
            images.append(sample[0])
            labels.append(sample[1])
    else:
        print("Incorrect model type in model attack. Please try again. Must be either PyTorch or TensorFlow.")
        sys.exit()

    # perform the attacks
    if attack_type == "uniform":
        attack = fb.attacks.L2AdditiveUniformNoiseAttack()
    elif attack_type == "gaussian":
        attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
    elif attack_type == "saltandpepper":
        attack = fb.attacks.SaltAndPepperNoiseAttack()
    elif attack_type == "boundary":
        attack = fb.attacks.BoundaryAttack()
    # NOTE: Doesn't look like spatial is being supported by the devs anymore, not sure if should be using
    elif attack_type == "spatial":
        attack = fb.attacks.SpatialAttack()
    elif attack_type == "deepfool":
        attack = fb.attacks.LinfDeepFoolAttack()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    accuracy_list = []
    print("Performing FoolBox Attacks for " + model_type + " with attack type " + attack_type)
    for i in tqdm(range(len(images))):
        raw_advs, clipped_advs, success = attack(fmodel, images[i], labels[i], epsilons=epsilons)
        if model_type == "pt":
            robust_accuracy = 1 - success.cpu().numpy().astype(float).flatten().mean(axis=-1)
        else:
            robust_accuracy = 1 - success.numpy().astype(float).flatten().mean(axis=-1)
        accuracy_list.append(robust_accuracy)
    return np.array(accuracy_list).mean()

@wandb_mixin
def multi_train(config):
    """Definition of side by side training of pytorch and tensorflow models, plus optional resiliency testing."""
    global NUM_CLASSES, MIN_RESILIENCY, MAX_DIFF, ONLY_CPU, MODEL_FRAMEWORK
    # print(NUM_CLASSES)
    if MODEL_FRAMEWORK == "pt":
        if ONLY_CPU:
            try:
                pt_test_acc, pt_model, pt_training_history, pt_val_loss, pt_val_acc = PT_MODEL(config, only_cpu=ONLY_CPU)
            except:
                print("WARNING: implementation not completed for using only CPU. Using GPU.")
                pt_test_acc, pt_model, pt_training_history, pt_val_loss, pt_val_acc = PT_MODEL(config)
        else:
            pt_test_acc, pt_model, pt_training_history, pt_val_loss, pt_val_acc = PT_MODEL(config)
        pt_model.eval()
        torch.save(pt_model.state_dict(), os.path.join(wandb.run.dir, "model.h5"))
        search_results = {'pt_test_acc': pt_test_acc}
        if not NO_FOOL:
            for attack_type in ['gaussian', 'deepfool']:
                pt_acc = model_attack(pt_model, "pt", attack_type, config, num_classes=NUM_CLASSES)
                search_results["pt" + "_" + attack_type + "_" + "accuracy"] = pt_acc
        pt_conv, pt_ave_conv_diff = found_convergence(pt_val_acc)
        # to avoid weird CUDA OOM errors
        del pt_model
        torch.cuda.empty_cache()
        search_results['pt_converged_bool'] = pt_conv
        search_results['pt_converged_average'] = pt_ave_conv_diff
        search_results['pt_training_loss'] = pt_training_history
        search_results['pt_validation_loss'] = pt_val_loss
        search_results['pt_validation_acc'] = pt_val_acc
        average_res = pt_test_acc
        data = [[x, y] for (x, y) in zip(list(range(len(pt_training_history))), pt_training_history)]
        table = wandb.Table(data=data, columns=["epochs", "training_loss"])
        wandb.log({"PT Training Loss": wandb.plot.line(table, "epochs", "training_loss", title="PT Training Loss")})
        data = [[x, y] for (x, y) in zip(list(range(len(pt_val_loss))), pt_val_loss)]
        table = wandb.Table(data=data, columns=["epochs", "validation loss"])
        wandb.log(
            {"PT Validation Loss": wandb.plot.line(table, "epochs", "validation loss", title="PT Validation Loss")})
        data = [[x, y] for (x, y) in zip(list(range(len(pt_val_acc))), pt_val_acc)]
        table = wandb.Table(data=data, columns=["epochs", "validation accuracy"])
        wandb.log({"PT Validation Accuracy": wandb.plot.line(table, "epochs", "validation accuracy",
                                                             title="PT Validation Accuracy")})
    else:
        if ONLY_CPU:
            try:
                tf_test_acc, tf_model, tf_training_history, tf_val_loss, tf_val_acc = TF_MODEL(config, only_cpu=ONLY_CPU)
            except:
                tf_test_acc, tf_model, tf_training_history, tf_val_loss, tf_val_acc = TF_MODEL(config)
        else:
            tf_test_acc, tf_model, tf_training_history, tf_val_loss, tf_val_acc = TF_MODEL(config)
        search_results = {}
        search_results['tf_test_acc'] = tf_test_acc
        tf_model.save(os.path.join(wandb.run.dir, "model.h5"))
        if not NO_FOOL:
            for attack_type in ['gaussian', 'deepfool']:
                pt_acc = model_attack(tf_model, "tf", attack_type, config, num_classes=NUM_CLASSES)
                search_results["tf" + "_" + attack_type + "_" + "accuracy"] = pt_acc
        tf_conv, tf_ave_conv_diff = found_convergence(tf_val_acc)
        search_results['tf_training_loss'] = tf_training_history
        search_results['tf_validation_loss'] = tf_val_loss
        search_results['tf_validation_acc'] = tf_val_acc
        search_results['tf_converged_bool'] = tf_conv
        search_results['tf_converged_average'] = tf_ave_conv_diff
        average_res = tf_test_acc
        data = [[x, y] for (x, y) in zip(list(range(len(tf_training_history))), tf_training_history)]
        table = wandb.Table(data=data, columns=["epochs", "training_loss"])
        wandb.log({"TF Training Loss": wandb.plot.line(table, "epochs", "training_loss", title="TF Training Loss")})
        data = [[x, y] for (x, y) in zip(list(range(len(tf_val_loss))), tf_val_loss)]
        table = wandb.Table(data=data, columns=["epochs", "validation loss"])
        wandb.log(
            {"TF Validation Loss": wandb.plot.line(table, "epochs", "validation loss", title="TF Validation Loss")})
        data = [[x, y] for (x, y) in zip(list(range(len(tf_val_acc))), tf_val_acc)]
        table = wandb.Table(data=data, columns=["epochs", "validation accuracy"])
        wandb.log({"TF Validation Accuracy": wandb.plot.line(table, "epochs", "validation accuracy",
                                                             title="TF Validation Accuracy")})
    # optimize simply for best test set accuracy
    search_results['average_res'] = average_res

    # log inidividual metrics to wanbd
    for key, value in search_results.items():
        wandb.log({key: value})
    # log custom training and validation curve charts to wandb
    try:
        tune.report(**search_results)
    except:
        print("Couldn't report Tune results. Continuing.")
        pass
    return search_results

def bitune_parse_arguments(args):
    """Parsing arguments specifically for bi tune experiments"""
    global PT_MODEL, TF_MODEL, NUM_CLASSES, NO_FOOL, MNIST, TRIALS, MAX_DIFF, FASHION, MIN_RESILIENCY
    global ONLY_CPU, OPTIMIZE_MODE, MODEL_TYPE, MAXIMIZE_CONVERGENCE, MODEL_FRAMEWORK
    if not args.model:
        print("NOTE: Defaulting to fashion dataset model training...")
        args.model = "fashion"
        NUM_CLASSES = 10
    else:
        if args.model == "caltech":
            PT_MODEL = caltech_pytorch_alexnet.caltech_pt_objective
            TF_MODEL = caltech_tensorflow_alexnet.caltech_tf_objective
            NUM_CLASSES = 102
            MODEL_TYPE = "caltech"
        elif args.model == "cinic":
            PT_MODEL = cinic_pytorch_alexnet.cinic_pt_objective
            TF_MODEL = cinic_tensorflow_alexnet.cinic_tf_objective
            NUM_CLASSES = 10
            MODEL_TYPE = "cinic"
        else:
            print("\n ERROR: Unknown model type. Please try again. "
                  "Must be one of: mnist, alexnet_cifar100, segmentation_cityscapes, or segmentation_gis.\n")
            sys.exit()
    if not args.trials:
        print("NOTE: Defaulting to 25 trials per scikit opt space...")
    else:
        TRIALS = int(args.trials)

    if args.only_cpu:
        ONLY_CPU = True

    if args.framework:
        MODEL_FRAMEWORK = args.framework

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    parser.add_argument('-n', '--start_space')
    parser.add_argument('-c', '--only_cpu', action='store_true')
    parser.add_argument('-p', '--project_name', default="hyper_sensitive")
    parser.add_argument('-f', '--framework', default='pt')
    args = parser.parse_args()
    bitune_parse_arguments(args)
    # print(PT_MODEL)
    # print(OPTIMIZE_MODE)
    spaceray.run_experiment(args, multi_train, ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/raylogs", cpu=8,
                                start_space=int(args.start_space), mode="max", project_name=args.project_name,
                                group_name='benchmark')