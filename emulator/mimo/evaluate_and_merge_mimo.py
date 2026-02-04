import os
import numpy as np
from joblib import dump, load
import pickle
from emulator.mimo.pyct.DataLoader import Cloud_Dataset, ToTensor
from emulator.mimo.pyct.cloud_unet_pl import CloudUNetPL
from mimo.models.mimo_unet import MimoUnetModel
from mimo.models.utils import repeat_subnetworks, compute_uncertainties
from emulator.mimo.pyct.viz import gen_plots
from emulator.mimo.pyct.utils import read_yaml, load_scalers, log_unscale_data, scaler_transform, log_scale_data
from emulator.mimo.pyct.generate_training_dataset import tiles2map, preprocess_single_file
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr

def main(config):

    # trim 25 % of tile size
    trim = int(config["tile_size"]/4)
    # set tiling stride to get overlap of size `trim`
    config['stride'] = config["tile_size"] - trim*2
    print(f"tile size = {config['tile_size']} pixel, trim = {trim} pixel (25%) --> set stride to {config['stride']} pixel.")

    out_dir = config["out_dir"]
    model_dir = config["model_dir"]
    os.makedirs(out_dir, mode = 0o777, exist_ok = True)
    plot_dir = config["plot_dir"]
    print(f"Using {plot_dir} to save figures")
    print(f"Reading saved model and scaler from {out_dir}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    input_scaler, output_scaler = load_scalers(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = None
    uid = None
    ind = -1
    for scene in config["scenes"]:
        ind += 1
        print(scene)
        uid = os.path.basename(os.path.splitext(scene)[0])
        out_subdir = os.path.join(out_dir, uid)
        print(out_subdir)
        os.makedirs(out_subdir, mode = 0o777, exist_ok = True)
        config["data_fname"] = scene

        input_fname  = config["input_fname"][ind]
        target_fname = config["target_fname"][ind]

        rad_tiled, target_tiled, *scene_shape = preprocess_single_file(input_fname, target_fname, config)

        if rad_tiled.ndim == 3:
            rad_tiled = np.expand_dims(rad_tiled,1)
        if target_tiled.ndim == 3:
            target_tiled = np.expand_dims(target_tiled,1)

        rad_tiled = rad_tiled.astype(np.float32)
        target_tiled = target_tiled.astype(np.float32)

        if config["log_scale_input"]:
            rad_tiled = log_scale_data(rad_tiled, True, config['fill_value'])
        if config["log_scale_target"]:
            target_tiled = log_scale_data(target_tiled, True, config['fill_value'])


        if input_scaler is not None:
            rad_tiled = scaler_transform(input_scaler, rad_tiled, method="transform")

        if output_scaler is not None:
            target_tiled = scaler_transform(output_scaler, target_tiled, method="transform")

        out_channels=1
        if config["multiview"]:
            out_channels=rad_tiled.shape[3]
        if unet is None:
            if config["use_mimo"]:
                unet = MimoUnetModel.load_from_checkpoint(os.path.join(model_dir, config["state_dict"]), map_location=device)
            else:
                unet = CloudUNetPL.load_from_checkpoint(os.path.join(model_dir, config["state_dict"]), map_location=device)
            #unet = UNet(in_channels=rad_tiled.shape[3], out_channels=out_channels, init_features=64)
            #unet.to(device=device)
            #state_dict = torch.load(os.path.join(model_dir, config["state_dict"]), map_location=device)
            #unet.load_state_dict(state_dict)
            #unet.eval()

        transformations = [transforms.ToTensor()]
        pred_transform = transforms.Compose(transformations)
 
        cloud_ds = Cloud_Dataset(rad_tiled.transpose(0,3,1,2), target_tiled.transpose(0,3,1,2), transform_image=pred_transform)
        dataloader = DataLoader(cloud_ds, batch_size=config["batch_size"], shuffle=False)
        output = np.zeros(target_tiled.shape, np.float32).transpose(0,3,1,2)
        alea_uncert = None
        epis_uncert = None
        log_params = None
        output_tmp = None
        if config["use_mimo"]:
            tmp_shape = (target_tiled.shape[0], unet.num_subnetworks, target_tiled.shape[3], target_tiled.shape[1], target_tiled.shape[2])
            log_params = torch.zeros(tmp_shape, dtype=torch.float32)
            output_tmp = torch.zeros(tmp_shape, dtype=torch.float32)
        with torch.no_grad():
            itr = 0
            for batch in tqdm(dataloader):
                X, y = batch["image"], batch["label"]
                X = X.to(device)
                if not config["use_mimo"]:
                    output[itr*config["batch_size"]:(itr+1)*config["batch_size"],:,:,:] = \
                            unet(X).detach().cpu().numpy()
                else:
                     x_rep = repeat_subnetworks(X, num_subnetworks=unet.num_subnetworks)
                     y_pred, log_param = unet(x_rep)
                     output_tmp[itr*config["batch_size"]:(itr+1)*config["batch_size"],:,:,:,:] = \
                             y_pred
                     log_params[itr*config["batch_size"]:(itr+1)*config["batch_size"],:,:,:,:] = \
                             log_param
                X = X.cpu()
                itr += 1

        if config["use_mimo"]:
            output, alea_uncert, epis_uncert = compute_uncertainties(unet.loss_fn, output_tmp, log_params)
            output = output.detach().cpu().numpy()
            alea_uncert = alea_uncert.detach().cpu().numpy()
            epis_uncert = epis_uncert.detach().cpu().numpy()
            output_tmp = output_tmp.detach().cpu().numpy()
            log_params = log_params.detach().cpu().numpy()


            #print(output.shape, target_tiled.shape, alea_uncert.shape, epis_uncert.shape)
        #print(output.shape, target_tiled.shape)
        output = output.transpose(0,2,3,1)
        if alea_uncert is not None:
            alea_uncert = alea_uncert.transpose(0,2,3,1)
            epis_uncert = epis_uncert.transpose(0,2,3,1)
        if config["plot_tiles"]:
            out_subdir_2 = os.path.join(out_subdir, "tiles_scaled")
            os.makedirs(out_subdir_2, mode = 0o777, exist_ok = True)
            gen_plots(rad_tiled, target_tiled, output, 0, ymin=target_tiled.min(),ymax=target_tiled.max(), \
                xmin=rad_tiled.min(), xmax=rad_tiled.max(),
                out_dir=out_subdir_2, log_scale = config["log_scale_target"], plot_metrics = True, 
                cloud_threshold= config["tau_min_threshold"], tile_size = config["metric_tile_size"],
                      angles=config["angles"], alea_uncert=alea_uncert, epis_uncert=epis_uncert)
 
        rad = tiles2map(rad_tiled, scene_shape[0], scene_shape[1], config["stride"], trim=trim)
        target = tiles2map(target_tiled, scene_shape[0], scene_shape[1], config["stride"], trim=trim)
        output_untiled = tiles2map(output, scene_shape[0], scene_shape[1], config["stride"], trim=trim)
        rad = np.expand_dims(rad, axis=0)
        target = np.expand_dims(target, axis=0)
        output_untiled = np.expand_dims(output_untiled, axis=0)
 
        alea_uncert_untiled = None
        epis_uncert_untiled = None
        if alea_uncert is not None:
            alea_uncert_untiled = tiles2map(alea_uncert, scene_shape[0], scene_shape[1], config["stride"], trim=trim)
            epis_uncert_untiled = tiles2map(epis_uncert, scene_shape[0], scene_shape[1], config["stride"], trim=trim)
            alea_uncert_untiled = np.expand_dims(alea_uncert_untiled, axis=0)
            epis_uncert_untiled = np.expand_dims(epis_uncert_untiled, axis=0)


            print(rad.shape, target.shape, output_untiled.shape, alea_uncert_untiled.shape, epis_uncert_untiled.shape)
        print(rad.shape, target.shape, output_untiled.shape)
        #gen_plots(output_untiled, target, rad, 
        #        uid, ymin=target.min(),ymax=target.max(),xmin=rad.min(), xmax=rad.max(), out_dir=out_subdir,
        #        log_scale = config["log_scale_target"], plot_metrics = True, 
        #        cloud_threshold= config["tau_min_threshold"], tile_size = config["metric_tile_size"],
        #          angles=config["angles"], alea_uncert=alea_uncert_untiled, epis_uncert=epis_uncert_untiled)

        #Handles single and multi-channel
        while rad.ndim < 4:
            rad = np.expand_dims(rad,0)
        if input_scaler is not None:
            rad = scaler_transform(input_scaler, rad, method="inverse_transform")
            rad_tiled = scaler_transform(input_scaler, rad_tiled, method="inverse_transform")

        if output_scaler is not None:
            target = scaler_transform(output_scaler, target, method="inverse_transform")
            target_tiled = scaler_transform(output_scaler, target_tiled, method="inverse_transform")
            output_untiled = scaler_transform(output_scaler, output_untiled, method="inverse_transform")
            output = scaler_transform(output_scaler, output, method="inverse_transform")

        if config["log_scale_input"]:
            rad = log_unscale_data(rad, True, config['fill_value'])
        if config["log_scale_target"]:
            target = log_unscale_data(target, True, config['fill_value'])
            output_untiled = log_unscale_data(output_untiled, True, config['fill_value'])

            target_tiled = log_unscale_data(target_tiled, True, config['fill_value'])
            output = log_unscale_data(output, True, config['fill_value'])

        """
        uid += "_unscaled"

        gen_plots(output_untiled, target, rad,
                uid, ymin=target.min(),ymax=target.max(),xmin=rad.min(), xmax=rad.max(), out_dir=out_subdir,
                log_scale = False, plot_metrics = True,
                cloud_threshold= np.exp(config["tau_min_threshold"]), tile_size = config["metric_tile_size"],
                  angles=config["angles"], alea_uncert=alea_uncert_untiled, epis_uncert=epis_uncert_untiled)


        if config["plot_tiles"]:
            out_subdir_3 = os.path.join(out_subdir, "tiles_unscaled")
            os.makedirs(out_subdir_3, mode = 0o777, exist_ok = True)
            gen_plots(output, target_tiled, rad,
                uid, ymin=target.min(),ymax=target.max(),xmin=rad.min(), xmax=rad.max(), out_dir=out_subdir_3,
                log_scale = False, plot_metrics = True,
                cloud_threshold= np.exp(config["tau_min_threshold"]), tile_size = config["metric_tile_size"],
                      angles=config["angles"], alea_uncert=alea_uncert, epis_uncert=epis_uncert) 
        """
    return output_untiled, epis_uncert_untiled, alea_uncert_untiled, scene_shape


if __name__ == '__main__':
    import argparse

    time = 10.5
    resolution = 40 # m
    #resolution = 280 # m

    if resolution == 280:
        configs = [
                    f"config/{time:.0f}h/config_nadir_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n261_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_261_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n456_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_456_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n60_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_60_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n705_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_705_mimo_single_angle_evaluate.yaml",
                ]
    elif resolution == 40:
        # configs = [
        #             f"config/{time:.0f}h/config_nadir_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_n261_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_261_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_n456_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_456_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_n60_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_60_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_n705_mimo_single_angle_evaluate_40m.yaml",
        #             f"config/{time:.0f}h/config_705_mimo_single_angle_evaluate_40m.yaml",
        #         ]
        configs = [
                    f"config/{time:.0f}h/config_nadir_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n261_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_261_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n456_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_456_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n60_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_60_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_n705_mimo_single_angle_evaluate.yaml",
                    f"config/{time:.0f}h/config_705_mimo_single_angle_evaluate.yaml",
                ]
    else:
        print("only 40 and 280 m implemented")

    y_pred = []
    alea = []
    epis = []
    vza = []
    for config_fname in configs:
        # config = read_yaml(config_fname)
        config = read_yaml(os.path.join('/Users/annajungbluth/Desktop/git/hybridCT/emulator/mimo', config_fname))
        y_hat_map, epis_uncert, alea_uncert, scene_shape = main(config)
        y_pred.append(y_hat_map)
        alea.append(alea_uncert)
        epis.append(epis_uncert)
        vza.append(config['angles'][0])

    dx = dy = config["resolution"]
    dims = ("vza", "x", "y")
    coords = {
        "x": np.arange(0, scene_shape[0]*dx, dx),
        "y": np.arange(0, scene_shape[1]*dx, dy),
        "vza": vza
    }
    da = xr.DataArray(np.array(y_pred)[:,0,:,:,0], coords=coords, dims=dims)
    ds = xr.Dataset({
        "tau_pre": da.transpose("x", "y", "vza"),
        "aleatoric_uncertainty": xr.DataArray(np.array(alea)[:,0,:,:,0], coords=coords, dims=dims).transpose("x","y","vza"),
        "epistemic_uncertainty": xr.DataArray(np.array(epis)[:,0,:,:,0], coords=coords, dims=dims).transpose("x","y","vza"),
        })

    save_dir = "../../data/predicted_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if resolution == 40:
        ds.to_netcdf(os.path.join(save_dir, f"MISR_{(dx*1e3):.0f}m_80x80km_{time:.1f}h_optical_thickness_predicted_MIMO_40m_toa.nc"))
    elif resolution == 280:
        ds.to_netcdf(os.path.join(save_dir, f"MISR_{(dx*1e3):.0f}m_80x80km_{time:.1f}h_optical_thickness_predicted_MIMO_toa.nc"))
    else:
        print("only 40 and 280 m implemented")
