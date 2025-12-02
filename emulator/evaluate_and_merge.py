from DataLoader import Cloud_Dataset, ToTensor
from Model import UNet
from torch.utils.data import DataLoader
import torch, yaml, os, joblib, numpy as np, xarray as xr, argparse
import utils
from generate_training_dataset import tiles2map, preprocess_global


def predict_for_config(cfg):
    out_dir = cfg["mount_dir"]
    save_fname = os.path.join(out_dir, utils.create_dataset_filename(cfg))
    if os.path.exists(save_fname):
        X, y, nx, ny = utils.read_preprocessed_tiles(cfg, save_fname)
    else:
        X, y, nx, ny = preprocess_global(cfg)

    X_scaled = utils.log_scale_data(X, True, cfg["fill_value"]) if cfg["log_scale_input"] else X
    inp_scaler = joblib.load(os.path.join(out_dir, "input_scaler.pkl"))
    tgt_scaler = joblib.load(os.path.join(out_dir, "output_scaler.pkl"))
    X_norm = utils.scaler_transform(inp_scaler, X_scaled, "transform")

    model = UNet(in_channels=X.shape[-1], out_channels=y.shape[-1], init_features=64)
    model.load_state_dict(torch.load(os.path.join(out_dir, "unet_ct.pth"),
                                    map_location="cpu"))
    model.eval()

    cloud = Cloud_Dataset(X_norm.transpose(0,3,1,2), y.transpose(0,3,1,2), ToTensor())
    dataloader = DataLoader(cloud, batch_size=X.shape[0], shuffle=False)
    for Xb, _ in dataloader:
        y_hat_norm = model(Xb).detach().numpy()

    y_hat = utils.scaler_transform(tgt_scaler, y_hat_norm.transpose(0,2,3,1), "inverse_transform")
    if cfg["log_scale_target"]:
        y_hat = utils.log_unscale_data(y_hat, cfg["fill_value"])

    y_map = tiles2map(y_hat, nx, ny, cfg["stride"])
    return y_map, nx, ny


def run_multiview(save_dir, time, res, tag=""):

    cameras = ["Aa","Af","Ba","Bf","Ca","Cf","Da","Df","An"]
    cfgs = [f"config/preprocessed_views{v}_res{res}m_val{time:.1f}h_eval.yaml"
            for v in cameras]

    y_pred, vza = [], []
    for config_path in cfgs:
        with open(config_path) as yml:
            cfg = yaml.safe_load(yml)
        y_map, nx, ny = predict_for_config(cfg)
        y_pred.append(y_map)
        vza.append(cfg["angles"][0])

    dx = dy = cfg["resolution"]
    coords = {
            "x": np.arange(0, nx*dx, dx),
            "y": np.arange(0, ny*dy, dy), "vza": vza,
            }
    da = xr.DataArray(np.array(y_pred)[...,0], coords=coords, dims=("vza","x","y"))
    xr.Dataset({"tau_pre": da.transpose("x","y","vza")}) \
            .to_netcdf(os.path.join(save_dir, f"MISR_{dx*1e3:.0f}m_80x80km_{time:.1f}h_optical_thickness_predicted{tag}.nc"))


def run_ablation(save_dir, time=10.5, res=280):
    sweep = {
            "sza": [20,30,40,50], # deg
            "wind": [5,10,20], # m/s
            }

    for param, vals in sweep.items():
        y_pred = []
        for v in vals:
            config_path = f"config/preprocessed_viewsAn_res{res}m_val{time:.1f}h_eval_ablation_{param.upper()}{v:02d}.yaml"
            with open(config_path) as yml:
                cfg = yaml.safe_load(yml)
            y_map, nx, ny = predict_for_config(cfg)
            y_pred.append(y_map)

        dx = dy = cfg["resolution"]
        coords = {
                param: vals,
                "x": np.arange(0, nx*dx, dx),
                "y": np.arange(0, ny*dy, dy),
                }
        da = xr.DataArray(np.array(y_pred)[...,0], coords=coords, dims=(param,"x","y"))
        xr.Dataset({"tau_pre": da.transpose("x","y",param)}) \
                .to_netcdf(os.path.join(save_dir, f"MISR_{dx*1e3:.0f}m_80x80km_{time:.1f}h_optical_thickness_predicted_ablation_{param}.nc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["multiview","ablation"],
                        default="multiview", help="prediction mode")
    mode = parser.parse_args().mode

    save_dir = "../data/predicted_data"
    os.makedirs(save_dir, exist_ok=True)
    if mode == "multiview":
        #run_multiview(save_dir, time=30.0, res=40, tag="_toa")
        run_multiview(save_dir, time=10.5, res=280, tag="")
    else:
        run_ablation(save_dir)
