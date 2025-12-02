import os
import argparse
import matplotlib.pyplot as plt
from skimage import measure
from scipy import stats
import numpy as np
import xarray as xr
import pandas as pd
import projection_utils as proj_utils
from custom_sart import custom_radon, custom_radon_ray_sum, iradon_sart_custom
from viz_rec import scatter_density

def crop_back(cloud, pad=10):
    left = np.where(cloud.sum(axis=0)>0)[0][0]
    right = np.where(cloud.sum(axis=0)>0)[0][-1]
    base = np.where(cloud.sum(axis=1)>0)[0][0]
    top = np.where(cloud.sum(axis=1)>0)[0][-1]
    slz = slice(base-pad, top+pad)
    slx = slice(left-pad, right+pad)
    return cloud[slz, slx], slz, slx

def get_beta_ext(reff=10., wvl=0.67, mie_table="~/libRadtran/data/wc/mie/wc.sol.mie.cdf"):
    """
    wvl in micron
    reff_value in micron, here homogeneous reff
    """
    #reff = np.ones_like(lwc) * reff_value
    for_what = f"for reff={reff}um, wvl={wvl}um"
    if os.path.exists(os.path.expanduser(mie_table)):
        optprop = xr.open_dataset(mie_table)
        ilam = np.where(optprop.wavelen==wvl)[0][0]
        ireff = np.where(optprop.reff==reff)[0][0]
        beta_ext = optprop.ext.sel(nlam=ilam, nreff=ireff).data
        print(f"Reading extinction from file {for_what}: {beta_ext} 1/km")
    elif reff==10. and wvl==0.67:
        beta_ext = 157.71496034205362 # source: libRadtran
        print(f"Setting extinction {for_what}: {beta_ext} 1/km")
    else:
        raise NotImplementedError(f"Provide optical properties {for_what}.")
    return beta_ext

def coarsen_resolution(dataset, res, coarse_res):
    """Return coarsened dataset based on original resolution `res`
    to coarse resolution `coarse_res`.
    """
    window_size = int(coarse_res/res)
    print(f"Coarsening dataset using window size {window_size}")
    coarse = dataset.coarsen(x=window_size, boundary='trim').mean().coarsen(y=window_size, boundary='trim').mean().coarsen(z=window_size, boundary='trim').mean()
    return coarse


def filter_labels(binary_image, props, min_size):
    filtered_props = []
    for prop in props:
        if prop.equivalent_diameter < min_size:
            binary_image[prop.slice[0], prop.slice[1]] = 0
        else:
            filtered_props.append(prop)
    label_image = measure.label(binary_image)
    props = measure.regionprops(label_image)
    return label_image, props


def generate_cloud_file_dataset(lwc_, reff_, z, dx, pad=0):
    nx, ny, nz = lwc_.shape

    # add padding
    lwc = np.zeros((nx+2*pad, ny+2*pad, nz))
    reff = np.ones((nx+2*pad, ny+2*pad, nz)) * 10
    slx = slice(pad,nx+pad)
    sly = slice(pad,ny+pad)
    slz = slice(pad,nz+pad)
    lwc[slx,sly,:] = lwc_.data
    nx, ny, nz = lwc.shape

    coords = {
            "x": np.arange(0, nx*dx, dx),
            "y": np.arange(0, ny*dx, dx),
            "z": z
            }
    x = np.arange(0, nx*dx, dx)[:nx]
    y = np.arange(0, ny*dx, dx)[:ny]
    z = z[:nz]
    ds = xr.Dataset({
        "lwc": xr.DataArray(lwc, dims=["x", "y", "z"], coords=[x, y, z]),
        "reff": xr.DataArray(reff, dims=["x", "y", "z"], coords=[x, y, z]),
        "delx": dx,
        "dely": dx,
        })
    return ds

def generate_cloud_file_netcdf(filename, lwc_, reff_, z, dx, pad=1):
    ds = generate_cloud_file_dataset(lwc_, reff_, z, dx, pad)
    ds.to_netcdf(filename)

def generate_cloud_file(filename, lwc, reff, z, dx):
    mystic_domain = {}
    cloud_params = {}
    nx, ny, nz = lwc.shape
    mystic_domain["nx"] = nx
    mystic_domain["ny"] = ny
    mystic_domain["nz"] = nz
    mystic_domain["dx"] = dx
    mystic_domain["dy"] = dx
    base_idx = np.where(np.sum(lwc, axis=(0,1)) > 0)[0][0]
    top_idx = np.where(np.sum(lwc, axis=(0,1)) > 0)[0][-1]
    cloud_params["z_base"] = z[base_idx]
    cloud_params["z_top"] = z[top_idx]
    cloud_params["z"] = z
    cloud_params["lwc"] = lwc
    cloud_params["reff"] = reff
    generate_cloud_file_shdom(filename, mystic_domain, cloud_params)


def compute_max_view_angle(cth1, cbh1, com1, cth2, cbh2, com2, width2):
    delta_x = com1 - (com2 + width2/2.)
    delta_z = cth2 - cbh1
    view_angle = np.rad2deg(np.arctan2(delta_x / delta_z))
    return view_angle

def save2netcdf(cloud_lwc, cloud_ext, multi_angle_views, angles, cloud_attrs, nc_fname):
    res = xr.Dataset(data_vars={"lwc": cloud_lwc, "ext": cloud_ext, "sinogram": multi_angle_views}, attrs=cloud_attrs)
    print("save 2 netcdf")
    res.to_netcdf(nc_fname)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predicted", "truth", "isolated"],
                        default="predicted", help="prediction mode")
    mode = parser.parse_args().mode

    if mode == "predicted": # use ML-predicted optical thickness field
        tag = ""
        print("Tomographic cloud reconstruction using predicted optical thickness field.")
    elif mode == "truth": # use ground-truth optical thickness field
        tag = "_trueCOT"
        print("Tomographic cloud reconstruction using ground-truth optical thickness field.")
    elif mode == "isolated": # use ground-truth optical thickness assuming an isolated cloud
        tag = "_radon"
        print("Tomographic cloud reconstruction using ground-truth optical thickness of isolated cloud (no neighbors).")

    plot = True
    plot_images = True
    time = 30.
    scene = f"wc_les_RICO_40m_80kmx80km_T_qc_{time:.1f}h"
    _save_dir = os.path.join("results", scene)
    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)
    data_path = "../data"
    if mode == "truth":
        misr = xr.open_dataset(os.path.join(data_path, f"ground_truth/MISR_40m_80x80km_{time:.1f}h_optical_thickness_toa.nc")).tau
    else:
        misr = xr.open_dataset(os.path.join(data_path, "predicted_data/MISR_40m_80x80km_30.0h_optical_thickness_predicted_toa.nc")).tau_pre
    dx_misr = 0.28
    with xr.open_dataset(os.path.join(data_path, f"ground_truth/{scene}.nc")) as les_:
        les = generate_cloud_file_dataset(les_.lwc.transpose("ny", "nx", "nz"), 10, les_.z, les_.dx, pad=0)
        lwc = les.lwc
        dx = les.delx.data # resolution in [m]

        # compute extinction
        beta_ext = get_beta_ext()

        lwc_proj = lwc.sum(axis=2)
        cloud_mask = lwc_proj > 0.2
        binary_image = cloud_mask.data
        # Label the objects in the image
        label_image_ = measure.label(binary_image)
        # Calculate properties of the labeled regions
        props_ = measure.regionprops(label_image_)
        min_diameter = 0.500 # [km]
        min_size = min_diameter / dx # pixels
        print(f"resolution {dx} km")
        label_image, props = filter_labels(binary_image, props_, min_size)
        for ir, region in enumerate(props[54:55]):
            save_dir = os.path.join(_save_dir, f"cloud_id{region.label:03d}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            slx, sly = region.slice
            if ir == 54:
                yloc = 20
            else:
                yloc = int(np.ceil((sly.stop - sly.start)/2.))
            # Extract the rectangular region from the original image
            cloud_lwc = lwc[slx, sly, :]
            cloud_ext = cloud_lwc * beta_ext
            cloud_levels = cloud_lwc.z.where(np.sum(cloud_lwc, axis=(0,1))>0).dropna("z").data
            cbh = cloud_levels.min()
            cth = cloud_levels.max()
            cloud_COM = (cbh + cth) / 2
            height = cth - cbh
            width = region.equivalent_diameter * dx
            width_x = (slx.stop - slx.start) * dx
            width_y = (sly.stop - sly.start) * dx
            aspect_ratio = width / height
            aspect_ratio_x = width_x / height
            aspect_ratio_y = width_y / height
            angles = np.sort(misr.vza.data)
            multi_angle_views = proj_utils.generate_views(misr, slx, sly, dx, cloud_COM=cloud_COM-120, angles=angles, offset=[-0.05,-0.04,-0.03,-0.01,0,0,0,0,0])

            pixel_offset_z = int(len(cloud_ext.z.data)//2 - np.argmin(abs(cloud_ext.z.data - cloud_COM)))
            _test_cloud = cloud_ext
            _test_cloud = _test_cloud.shift(z=pixel_offset_z, fill_value=0)
            # Pad test cloud to square
            nx, ny, nz = _test_cloud.shape
            nx_new = ny_new = nz_new = max(nx, ny, nz)
            offset_x = int((nx_new-nx)/2)
            offset_y = int((ny_new-ny)/2)
            offset_z = int((nz_new-nz)/2)
            pad_x = (offset_x, nx_new-nx-offset_x)
            pad_y = (offset_y, ny_new-ny-offset_y)
            pad_z = (offset_z, nz_new-nz-offset_z)
            test_cloud = _test_cloud.pad({"x": pad_x, "z":pad_z}, constant_values=0)
            new_coords = np.concatenate([
                _test_cloud.x.data[0] + np.arange(-pad_x[0], 0) * dx,  # Before padding
                _test_cloud.x.data,                                      # Original coordinates
                _test_cloud.x.data[-1] + np.arange(1, pad_x[1] + 1) * dx  # After padding
            ])
            test_cloud = test_cloud.assign_coords({"x": new_coords})
            rotation_center = test_cloud.data.T.shape[0]//2
            # Compute sinogram of test cloud
            sino_test = custom_radon(test_cloud.data.T[:,yloc,:], theta=angles, resolution=dx)

            _sino_proj = multi_angle_views[:,:,::-1]
            # test sinogram
            nx, ny, nz = _sino_proj.shape
            nx_new, nz_new = sino_test.shape
            offset_x = int((nx_new-nx)/2)
            offset_z = int((nz_new-nz)/2)
            pad_x = (offset_x, nx_new-nx-offset_x)
            pad_z = (offset_z, nz_new-nz-offset_z)
            _sino_proj_pad = _sino_proj.pad({"x": pad_x, "z":pad_z}, constant_values=0)
            new_coords = np.concatenate([
                _test_cloud.x.data[0] + np.arange(-pad_x[0], 0) * dx,  # Before padding
                _test_cloud.x.data,                                      # Original coordinates
                _test_cloud.x.data[-1] + np.arange(1, pad_x[1] + 1) * dx  # After padding
            ])
            sino_proj_pad = _sino_proj_pad.assign_coords({"x": new_coords})

            # Tomographic reconstruction of projected MYSTIC data
            data_test = test_cloud.data.T
            data_rec_proj = np.zeros_like(data_test)
            niter = 100
            for i in range(sino_proj_pad.shape[1]):
                mask = data_test[:,i] == 0
                prior_proj = np.zeros_like(data_test[:,i])
                if mode == "isolated":
                    sino_direct = custom_radon(test_cloud.data.T[:,i,:], theta=angles, resolution=dx)
                for iiter in range(niter):
                    if mode == "isolated":
                        sl_rec = iradon_sart_custom(sino_direct, theta=angles, image=prior_proj, resolution=dx)
                    else:
                        sl_rec = iradon_sart_custom(sino_proj_pad.data[:,i,:], theta=angles, image=prior_proj, resolution=dx)
                    # apply cloud mask
                    sl_rec[mask] = 0
                    # clip negative results to zero
                    sl_rec[sl_rec<0] = 0
                    # use current estimate as prior
                    prior_proj = sl_rec
                data_rec_proj[:,i,:] = sl_rec

            error_proj = data_rec_proj - data_test
            r, _ = stats.pearsonr(data_test.ravel(), data_rec_proj.ravel())
            try:
                slope, offset = np.polyfit(data_rec_proj.ravel(), data_test.ravel(), 1)
            except:
                slope = 0
                offset = 0
            vmin = data_test.min()
            vmax = data_test.max()
            vlim = max(abs(vmin), abs(vmax))
            cmap = "viridis"
            if plot:
                # crop back before plotting
                _, crop_slx, crop_slz = crop_back(data_test[:,yloc])
                crop_slx = slice(offset_z, data_test[:,yloc].shape[0]-offset_z)
                true_slice = data_test[:,yloc][crop_slx, crop_slz]
                pred_slice = data_rec_proj[:,yloc][crop_slx, crop_slz]
                error_slice = error_proj[:,yloc][crop_slx, crop_slz]
                mask_slice = true_slice > 0
                # get grid in km
                nx_sl, nz_sl = true_slice.shape
                x_sl = np.arange(0, nx_sl*dx, dx)
                z_sl = np.arange(0, nz_sl*dx, dx)
                xgrid, zgrid = np.meshgrid(x_sl, z_sl-z_sl.max()/2., indexing="ij")

                fig, ax = plt.subplots(1, 3, figsize=(6, 3), dpi=200, sharex=True, sharey=True, constrained_layout=True)
                ax[0].set_title("Truth")
                im0 = ax[0].pcolormesh(zgrid, xgrid, np.ma.masked_array(true_slice, ~mask_slice), cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(im0, ax=ax[0], shrink=0.8)
                ax[1].set_title("Reconstruction")
                im1 = ax[1].pcolormesh(zgrid, xgrid, np.ma.masked_array(pred_slice, ~mask_slice), cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(im1, ax=ax[1], shrink=0.8)
                ax[2].set_title("Reconstruction error")
                im2 = ax[2].pcolormesh(zgrid, xgrid, np.ma.masked_array(error_slice, ~mask_slice), cmap="RdBu_r", vmin=-vlim, vmax=vlim)
                ax[2].contour(zgrid, xgrid, ~mask_slice, levels=[0], colors="k", linewidths=1)
                cbar = plt.colorbar(im2, ax=ax[2], shrink=0.8)
                cbar.set_label(r"[km$^{-1}$]")
                rmse = np.sqrt(np.nanmean((pred_slice - true_slice)**2))
                plt.suptitle(f"Reconstruction cloud id={region.label}, RMSE={rmse:.2f}"+r" km$^{-1}$"+f", yloc={yloc}")
                ax[0].set_ylabel("z [km]")
                for iax in range(3):
                    ax[iax].set_xlabel("x [km]")
                    ax[iax].set_aspect("equal")
                plt.savefig(os.path.join(save_dir, f"reconstruction_results_yloc{yloc:03d}_grid{tag}.png"))
                plt.show()


            # crop x and y-axis tight
            left = np.where(data_test.sum(axis=0)>0)[0][0]
            right = np.where(data_test.sum(axis=0)>0)[0][-1]
            base = np.where(data_test.sum(axis=1)>0)[0][0]
            top = np.where(data_test.sum(axis=1)>0)[0][-1]
            true_profile = np.ma.masked_array(true_slice, ~(true_slice>0)).mean(axis=1)
            true_profile.data[true_profile.mask] = 0
            pred_profile = np.ma.masked_array(pred_slice, true_slice==0).mean(axis=1)
            pred_profile.data[pred_profile.mask] = 0
            rmse_profile = np.sqrt(np.mean((pred_profile.data - true_profile.data)**2))
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), dpi=200, constrained_layout=True)
                ax[0].plot(true_profile.data, xgrid[:,0], label="True", c="k")
                ax[0].plot(pred_profile.data, xgrid[:,0], label="Reconstructed", c="C0")
                ax[0].set_title(f"Vertical Profile, RMSE={rmse_profile:.1f}"+r" km$^{-1}$")
                ax[0].legend()
                ax[0].set_xlabel(r"Extinction [km$^{-1}$]")
                ax[0].set_ylabel("Altitude [km]")

                ax[1] = scatter_density(ax[1], data_test.ravel(), data_rec_proj.ravel(), n_sample=len(data_test.ravel()), x_name="True", y_name="Reconstructed", stratified=True)
                ax[1].plot([0, np.nanmax(data_test)], [0, np.nanmax(data_test)], "k--")
                ax[1].plot([vmin, vmax], [vmin*slope+offset, vmax*slope+offset], "C0--", label=f"linear fit: {slope:.2f}x + {offset:.2f}", zorder=200)
                ax[1].set_xlim(0, np.nanmax(data_test))
                ax[1].set_ylim(0, np.nanmax(data_test))
                ax[1].set_title(fr"R$^2$ = {r:.2f}")
                ax[1].legend()
                ax[1].set_xlabel(r"True Extinction [km$^{-1}$]")
                ax[1].set_ylabel(r"Rec. Extinction [km$^{-1}$]")
                plt.savefig(os.path.join(save_dir, f"scatter_profile_results{tag}.png"))
                plt.show()

