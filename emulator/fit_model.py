import os
import argparse
from joblib import dump, load
import pickle
import matplotlib
matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm
from functools import partial
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import matplotlib.pyplot as plt

from DataLoader import Cloud_Dataset, ToTensor
from torch.utils.data import DataLoader
from Model import UNet
from generate_training_dataset import preprocess_global
from evaluation_utils import rmse
from viz import gen_plots
import utils

def evaluate_loss_gpu(net, data_iter, device=None, plot=False, log_scale = False, 
        tune=False, ymin=-10, ymax=10, xmin=0, xmax=10, out_dir=".", cloud_threshold=np.inf, metric_tile_size = 10, angles=[0]):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        uid = 0
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y_hat = net(X)
            y = y.to(device)
            l = rmse(y_hat, y)
            if plot:
                if "cuda" in device.type:
                    y = y.detach().cpu().numpy()
                    y_hat = y_hat.detach().cpu().numpy()
                    X = X.detach().cpu().numpy()
                gen_plots(y_hat, y, X, uid, ymin, ymax, xmin, xmax, out_dir, log_scale, True, cloud_threshold, metric_tile_size, angles)
                uid += y.shape[0]
    loss = l/X.shape[0]
    print(f"loss = {l}, size y={y.shape[0]}")
    return loss

def train(net, train_iter, test_iter, val_iter, config, device, ymin,ymax,xmin,xmax):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    #writer = SummaryWriter(comment=f'cloud_ct_tb')
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if hasattr(config["learning_rate"], '__iter__'):
        lr = config["learning_rate"][0]
    else:
        lr = config["learning_rate"]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss = nn.MSELoss()
    num_batches = len(train_iter)
    val_loss = 0.0
    for epoch in range(config["epochs"]):
        # Sum of training loss, sum of training accuracy, no. of examples
        net.train()
        i = 0
        for (X, y) in tqdm(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss = rmse(y_hat, y)
                #print(f"i={i} loss={l},", train_loss)
                if i % 10 == 0:
                    #writer.add_scalar('Loss/train', train_loss.item(), global_step=epoch * len(train_iter) + i)
                    val_loss = evaluate_loss_gpu(net, val_iter, device, False, config["log_scale_target"], angles=config["angles"])
                    #writer.add_scalar('Loss/validation', val_loss.item(), global_step=epoch * len(train_iter) + i)
            i += 1

        if config["run_tune"]:
            with ray.tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)
            ray.tune.report(loss=val_loss.detach().cpu().numpy())

        plot = False
        if epoch == config["epochs"]-1:
            plot = config["plot_test_images"]
        test_loss = evaluate_loss_gpu(net, test_iter, device, plot, config["log_scale_target"],config["run_tune"],
                ymin, ymax, xmin, xmax, config["out_dir"], config["tau_min_threshold"], config["metric_tile_size"], angles=config["angles"])
        #writer.add_scalar('Loss/test', test_loss.item(), epoch)
        print(f"epoch={epoch} loss={test_loss}")
    #writer.flush()
    #writer.close()


def load_and_preprocess_train(config):
    if config["load_and_scale"]:
        train_test_val_dataset = utils.load_train_test_val_dataset(config['out_dir'])
    else:
        input_tiled, target_tiled, nx, ny = preprocess_global(config)
        # save results to netCDF
        #utils.save2dataset(input_tiled, target_tiled, nx, ny, config)
        print(f"rad tiled {input_tiled.shape}")
        input_tiles_scaled, target_tiles_scaled = utils.scale_training_data(input_tiled, target_tiled, config)
        # split test/train/val
        train_test_val_dataset = utils.preprocess_training_data(input_tiles_scaled.transpose(0,3,1,2), target_tiles_scaled.transpose(0,3,1,2), config)
    print(f"train shape {train_test_val_dataset[0].shape}")
    return train_test_val_dataset


def main(config):

    out_dir = config["out_dir"]
    os.makedirs(out_dir, mode = 0o777, exist_ok = True) 

    x_train, x_test, x_val, y_train, y_test, y_val = load_and_preprocess_train(config)
    print(f"shape before training: X {x_train.shape}, y {y_train.shape}")
    cloud_train = Cloud_Dataset(x_train, y_train, transform=ToTensor())
    train_dataloader = DataLoader(cloud_train, batch_size=config["batch_size"], shuffle=True)
    cloud_val = Cloud_Dataset(x_val, y_val, transform=ToTensor())
    val_dataloader = DataLoader(cloud_val, batch_size=config["batch_size"], shuffle=True)
    cloud_test = Cloud_Dataset(x_test, y_test, transform=ToTensor())
    test_dataloader = DataLoader(cloud_test, batch_size=config["batch_size"], shuffle=True) 

    unet = UNet(in_channels=x_train.shape[1], out_channels=y_train.shape[1], init_features=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(unet, train_dataloader, test_dataloader, val_dataloader, config, device, y_test.min(), y_test.max(),x_test.min(), x_test.max())

    state_dict = unet.state_dict()
    torch.save(state_dict, os.path.join(out_dir, config["state_dict"]))



def tune_model(config):

    config["log_scale_input"] = ray.tune.choice([True, False])
    config["log_scale_target"] = ray.tune.choice([True, False])
    config["normalize_input"] = ray.tune.choice([True, False])
    config["normalize_target"] = ray.tune.choice([True, False])

    config["batch_size"] = ray.tune.choice([32, 64, 128])
    config["learning_rate"] = ray.tune.loguniform(1e-4, 1e-1),
    config["epochs"] = ray.tune.choice([10, 50, 100, 150]) 

    if config["tune_tiles"]:
        config["stride"] = ray.tune.choice([int(round(config["stride"] / 2)), config["stride"], int(config["stride"] * 2), int(config["stride"] * 4)])
        config["patch_size"] = ray.tune.choice([config["patch_size"], int(config["patch_size"] * 2), int(config["patch_size"] * 4)])

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=150,
        grace_period=config["grace_period"],
        reduction_factor=config["reduction_factor"])

    reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"], metric="loss", mode="min", sort_by_metric=True) 

    result = ray.tune.run(
        partial(main),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        local_dir = config['out_dir'],
        num_samples=config["num_samples"],
        raise_on_failed_trial = False,
        keep_checkpoints_num=2,
        max_concurrent_trials = 2,
        progress_reporter=reporter)

    best_trial = result.get_best_trial(metric="loss", mode ="min", scope="last", filter_nan_and_inf=True)
    print(best_trial, "HERE BEST TRIAL")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    with open(config["out_config"], "w", encoding = "utf-8") as yaml_file:
        dump = pyyaml.dump(best_trial.config, default_flow_style=False, allow_unicode=True, encoding=None)
        yaml_file.write(dump)

    return best_trial.config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml")
    args = parser.parse_args()
    config = utils.read_yaml(args.yaml)
    if config["run_tune"]:
        ray.init()
        try:
            config = tune_model(config)
        finally:
            ray.shutdown()

    main(config)

