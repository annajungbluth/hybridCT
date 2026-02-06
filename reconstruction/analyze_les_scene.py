import pickle
import os
import argparse
import matplotlib.pyplot as plt
from networkx import volume
from skimage import measure
from scipy import stats
import numpy as np
import xarray as xr
import pandas as pd
import projection_utils as proj_utils
from custom_sart import custom_radon, custom_radon_ray_sum, iradon_sart_custom
from viz_rec import scatter_density
from datetime import datetime
import plotly.graph_objects as go

# NOTE: Check that this import works correctly
from analyze_mystic_scene import get_beta_ext, filter_labels, generate_cloud_file_dataset
from vipct_utils import generate_cloud_file_dataset_vipct, extract_viewing_angles, rectify_with_projection_matrix, create_aligned_views, coarsen_image_resolution, upsample_volume

"""
analyze_les_scene.py
Utility script to load LES cloud scenes and perform tomographic
reconstruction experiments using predicted or ground-truth
optical thickness (COT) fields from either HybridCT or VIPCT. 
Produces diagnostics and plots comparing reconstructed extinction to truth.
"""

MISR_ANGLES = np.array([-70.5, -60.0, -45.6, -26.1, 0.0, 26.1, 45.6, 60.0, 70.5]) # MISR view angles in degrees
VIPCT_ANGLES = np.array([-39.3, -31.27, -21.87, -11.26, 0.13, 11.43, 22.03, 31.40, 39.41, 46.13])

def crop_back(cloud, pad=10):
    """
    Function to crop back to original cloud extent plus some padding.
    Parameters:
    cloud : numpy.ndarray
        2D cloud slice array
    pad : int
        Number of pixels to pad around the cropped region
    
    Returns:
    cropped_cloud : numpy.ndarray
        Cropped 2D cloud slice array
    slz : slice
        Slice object for z-dimension used for cropping
    slx : slice
        Slice object for x-dimension used for cropping
    """
    # find non-empty columns
    left = np.where(cloud.sum(axis=0)>0)[0][0] # leftmost non-empty column
    right = np.where(cloud.sum(axis=0)>0)[0][-1] # rightmost non-empty column
    base = np.where(cloud.sum(axis=1)>0)[0][0] # bottommost non-empty row
    top = np.where(cloud.sum(axis=1)>0)[0][-1] # topmost non-empty row
    slz = slice(base-pad, top+pad) # add some padding
    slx = slice(left-pad, right+pad) # add some padding
    return cloud[slz, slx], slz, slx # return cropped cloud and slices

def get_cloud_mask(ds, threshold=0):
    """
    Function to create a 2D cloud mask 

    Parameters:
    ds : xarray.DataArray
        3D cloud data array (nx, ny, nz)    
    threshold : float   
        Threshold value to create binary cloud mask

    Returns:
    binary_image : numpy.ndarray
        2D binary cloud mask (nx, ny)
    """
    # create 2D cloud mask by vertically summing volumes in the scene
    ds_proj = ds.sum(axis=2) # this should be shape (nx, ny)
    cloud_mask = ds_proj > threshold # threshold to create binary cloud mask, still in dataset format
    binary_image = cloud_mask.data # convert to numpy array
    return binary_image

def get_cloud_properties(data, slx=None, sly=None, beta_ext=None):
    """
    Function to extract cloud properties from 3D cloud data
    Parameters:
    data : xarray.DataArray
        3D cloud data array (nx, ny, nz)
    slx : slice, optional
        Slice object for x-dimension to extract cloud region
    sly : slice, optional
        Slice object for y-dimension to extract cloud region
    beta_ext : xarray.DataArray, optional
        Extinction coefficient array to compute cloud extinction

    Returns:
    cloud_dict : dict
        Dictionary containing cloud properties:
        - 'cloud_data': 3D cloud data array for the selected region
    """
    cloud_dict = {}
    if slx is not None and sly is not None:
        cloud_data = data[slx, sly, :]
    else:
        cloud_data = data
    cloud_dict['cloud_data'] = cloud_data

    if beta_ext is not None:
        cloud_ext = cloud_data * beta_ext
        cloud_dict['cloud_ext'] = cloud_ext
    else:
        cloud_dict['cloud_ext'] = cloud_data

    # get cloud levels in z with non-zero values
    cloud_levels = cloud_data.z.where(np.sum(cloud_data, axis=(0,1))>0).dropna("z").data 
    cbh = cloud_levels.min() # cloud base height
    cth = cloud_levels.max() # cloud top height
    ccom = (cbh + cth) / 2 # cloud center of mass height
    height = cth - cbh # cloud height

    cloud_dict['cbh'] = cbh
    cloud_dict['cth'] = cth
    cloud_dict['center_of_mass'] = ccom
    cloud_dict['height'] = height

    return cloud_dict

def get_test_cloud(cloud_dict, dx, dz):
    """
    Function to extract and pad the test cloud volume from cloud dictionary
    Parameters:
    cloud_dict : dict
        Dictionary containing cloud properties including 'cloud_ext' DataArray
    dx : float
        Spatial resolution in x-direction [km]
    dz : float
        Spatial resolution in z-direction [km]

    Returns:
    test_cloud : xarray.DataArray
        Padded 3D cloud extinction data array
    """
    test_cloud = cloud_dict['cloud_ext']
    pixel_offset_z = int(len(test_cloud.z.data)//2 - np.argmin(abs(test_cloud.z.data - cloud_dict['center_of_mass']))) # shift cloud in z to center around COM
    test_cloud = test_cloud.shift(z=pixel_offset_z, fill_value=0) # shift cloud in z to center around COM

    nx, ny, nz = test_cloud.shape
    nx_new = ny_new = nz_new = max(nx, ny, nz) # new size to pad to (make cube), e.g. (23, 37, 66) -> (66, 66, 66)

    # no y-padding needed since we are doing 2D reconstructions in x-z slices and y is the projection dimension
    offset_x = int((nx_new-nx)/2) # compute offsets for padding, e.g. (66-23)/2 = 21.5 -> 21
    offset_z = int((nz_new-nz)/2) # compute offsets for padding, e.g. (66-66)/2 = 0

    pad_x = (offset_x, nx_new-nx-offset_x) # padding for x, e.g. (21, 22)
    pad_z = (offset_z, nz_new-nz-offset_z) # padding for z, e.g. (0, 0)

    test_cloud_padded = test_cloud.pad({"x": pad_x, "z":pad_z}, constant_values=0) # pad in x and z with zeros, new shape should be (66, 37, 66)

    new_x_coords = np.concatenate([
        test_cloud.x.data[0] + np.arange(-pad_x[0], 0) * dx,  # Before padding
        test_cloud.x.data,                                      # Original coordinates
        test_cloud.x.data[-1] + np.arange(1, pad_x[1] + 1) * dx  # After padding
    ])
    new_z_coords = np.concatenate([
        test_cloud.z.data[0] + np.arange(-pad_z[0], 0) * dz,  # Before padding
        test_cloud.z.data,                                      # Original coordinates  
        test_cloud.z.data[-1] + np.arange(1, pad_z[1] + 1) * dz  # After padding
    ])
    test_cloud_padded = test_cloud_padded.assign_coords({"x": new_x_coords, "z": new_z_coords}) # update coordinates after padding

    return test_cloud, test_cloud_padded # shape (nx_new, ny_new, nz_new)

def get_padded_sinogram(sinogram, test_cloud, test_cloud_padded, angles, dx):
    """
    Function to pad sinogram projections to match reconstruction size
    Parameters:
    sinogram : xarray.DataArray
        3D sinogram data array (nx, ny, n_angles)
    test_cloud : xarray.DataArray
        Original 3D cloud extinction data array before padding
    test_cloud_padded : xarray.DataArray
        Padded 3D cloud extinction data array
    angles : numpy.ndarray
        Array of projection angles in degrees
    dx : float
        Spatial resolution in [km]

    Returns:
    xarray.DataArray
        Padded sinogram data array
    """

    # adjust shapes for reconstruction
    nx, ny, nz = sinogram.shape # nx = 23/116, ny = 37/116, nz = 9/10
    nx_new, nz_new = test_cloud_padded.data.T.shape[0], len(angles) # nx_new = 66/116, nz_new = 9/10

    offset_x = int((nx_new-nx)/2) # compute offsets for padding, e.g. (66-23)/2 = 21.5 -> 21
    offset_z = int((nz_new-nz)/2) # compute offsets for padding, e.g. (9-9)/2 = 0
    pad_x = (offset_x, nx_new-nx-offset_x) # padding for x, e.g. (21, 22)
    pad_z = (offset_z, nz_new-nz-offset_z) # padding for z, e.g. (0, 0)

    # pad sinogram projections to match reconstruction size
    sinogram_padded = sinogram.pad({"x": pad_x, "z":pad_z}, constant_values=0) # pad in x and z with zeros, new shape should be (66, 37, 9)
    new_coords = np.concatenate([
        test_cloud.x.data[0] + np.arange(-pad_x[0], 0) * dx,  # Before padding
        test_cloud.x.data,                                      # Original coordinates
        test_cloud.x.data[-1] + np.arange(1, pad_x[1] + 1) * dx  # After padding
    ])
    # sino_proj_pad is basically an x-padded version of the different camera views
    sinogram_padded = sinogram_padded.assign_coords({"x": new_coords})

    return sinogram_padded, offset_z

def get_sart_reconstruction(sinogram_padded, test_cloud_padded, angles, dx, niter=200):
    """
    Wrapper function to perform SART reconstruction on padded sinogram data
    Parameters:
    sinogram_padded : xarray.DataArray
        Padded sinogram data array
    test_cloud_padded : xarray.DataArray
        Padded 3D cloud extinction data array
    angles : numpy.ndarray
        Array of projection angles in degrees
    dx : float
        Spatial resolution in [km]
    niter : int
        Number of SART iterations
    
    Returns:
    reconstruction : numpy.ndarray
        Reconstructed 3D cloud extinction data array
    """
    test_cloud_data = test_cloud_padded.data.T # from (x, y, z) to (z, y, x) for reconstruction, shape (nz_new, ny_new, nx_new)

    reconstruction = np.zeros_like(test_cloud_data) # array to hold reconstructed data

    for y in range(sinogram_padded.shape[1]): # iterate over each projection, i.e. each y-location
        mask = test_cloud_data[:,y] == 0 # shape = (66, 66), i.e. (nz, nx), mask of where the true extinction is zero
        prior = np.zeros_like(test_cloud_data[:,y]) # initial prior for SART reconstruction, shape = (66, 66) of zeroes
        for iiter in range(niter):
            sl_rec = iradon_sart_custom(sinogram_padded.data[:,y,:], theta=angles, image=prior, resolution=dx)
            # apply cloud mask
            sl_rec[mask] = 0 # this mask comes from the ground truth extinction field and would not be available in practice
            # clip negative results to zerotes
            sl_rec[sl_rec<0] = 0
            # use current estimate as prior
            prior = sl_rec
        reconstruction[:,y,:] = sl_rec

    return reconstruction

def plot_results(test_cloud_3d, reconstruction, error, yloc, offset_z, dx, save_dir, tag="", cmap="viridis"):
        
        R, _ = stats.pearsonr(test_cloud_3d.ravel(), reconstruction.ravel())
        try:
            slope, offset = np.polyfit(reconstruction.ravel(), test_cloud_3d.ravel(), 1)
        except:
            slope, offset = 0, 0

        # NOTE: Check if this is needed for HybridCT
        # # crop back to the original cloud extent plus some padding
        # _, _, crop_slz = crop_back(test_cloud_3d[:,yloc])
        # crop_slx = slice(offset_z, test_cloud_3d[:,yloc].shape[0]-offset_z)

        # # extract slices for plotting
        # true_slice = test_cloud_3d[:,yloc][crop_slx, crop_slz]
        # mask_slice = true_slice > 0

        # pred_slice = reconstruction[:,yloc][crop_slx, crop_slz]
        # error_slice = error[:,yloc][crop_slx, crop_slz]

        # extract slices for plotting
        true_slice = test_cloud_3d[:,yloc]
        mask_slice = true_slice > 0

        pred_slice = reconstruction[:,yloc]
        error_slice = error[:,yloc]

        # get grid in [km]
        nx_sl, nz_sl = true_slice.shape
        x_sl = np.arange(0, nx_sl*dx, dx)
        z_sl = np.arange(0, nz_sl*dx, dx)
        xgrid, zgrid = np.meshgrid(x_sl, z_sl-z_sl.max()/2., indexing="ij")

        # get plotting extents
        vmin = test_cloud_3d.min()
        vmax = test_cloud_3d.max()
        vlim = max(abs(vmin), abs(vmax))

        # plot results
        fig, ax = plt.subplots(1, 3, figsize=(6, 3), dpi=200, sharex=True, sharey=True, constrained_layout=True)
        ax[0].set_title("Truth")
        im0 = ax[0].pcolormesh(zgrid, xgrid, np.ma.masked_array(true_slice, ~mask_slice), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im0, ax=ax[0], shrink=0.8)
        ax[1].set_title("Reconstruction")
        im1 = ax[1].pcolormesh(zgrid, xgrid, np.ma.masked_array(pred_slice, ~mask_slice), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax[1], shrink=0.8)
        ax[2].set_title("Reconstruction Error")
        im2 = ax[2].pcolormesh(zgrid, xgrid, np.ma.masked_array(error_slice, ~mask_slice), cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax[2].contour(zgrid, xgrid, ~mask_slice, levels=[0], colors="k", linewidths=1)
        cbar = plt.colorbar(im2, ax=ax[2], shrink=0.8)
        cbar.set_label(r"[km$^{-1}$]")
        rmse = np.sqrt(np.nanmean((pred_slice - true_slice)**2))
        plt.suptitle(f"Reconstruction, RMSE={rmse:.2f}"+r" km$^{-1}$"+f", yloc={yloc}")
        ax[0].set_ylabel("z [km]")
        for iax in range(3):
            ax[iax].set_xlabel("x [km]")
            ax[iax].set_aspect("equal")
        plt.savefig(os.path.join(save_dir, f"reconstruction_results_yloc{yloc:03d}{tag}.png"))
        plt.show()


        # crop x and y-axis tight
        true_profile = np.ma.masked_array(true_slice, ~(true_slice>0)).mean(axis=1)
        true_profile.data[true_profile.mask] = 0
        pred_profile = np.ma.masked_array(pred_slice, true_slice==0).mean(axis=1)
        pred_profile.data[pred_profile.mask] = 0
        rmse_profile = np.sqrt(np.mean((pred_profile.data - true_profile.data)**2))

        fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), dpi=200, constrained_layout=True)
        ax[0].plot(true_profile.data, xgrid[:,0], label="True", c="k")
        ax[0].plot(pred_profile.data, xgrid[:,0], label="Reconstructed", c="C0")
        ax[0].set_title(f"Vertical Profile, RMSE={rmse_profile:.1f}"+r" km$^{-1}$")
        ax[0].legend()
        ax[0].set_xlabel(r"Extinction [km$^{-1}$]")
        ax[0].set_ylabel("Altitude [km]")

        ax[1] = scatter_density(ax[1], test_cloud_3d.ravel(), reconstruction.ravel(), n_sample=len(test_cloud_3d.ravel()), x_name="True", y_name="Reconstructed", stratified=True)
        ax[1].plot([0, np.nanmax(test_cloud_3d)], [0, np.nanmax(test_cloud_3d)], "k--")
        ax[1].plot([vmin, vmax], [vmin*slope+offset, vmax*slope+offset], "C0--", label=f"linear fit: {slope:.2f}x + {offset:.2f}", zorder=200)
        ax[1].set_xlim(0, np.nanmax(test_cloud_3d))
        ax[1].set_ylim(0, np.nanmax(test_cloud_3d))
        ax[1].set_title(fr"R$^2$ = {R:.2f}")
        ax[1].legend()
        ax[1].set_xlabel(r"True Extinction [km$^{-1}$]")
        ax[1].set_ylabel(r"Rec. Extinction [km$^{-1}$]")
        plt.savefig(os.path.join(save_dir, f"scatter_profile_results{tag}.png"))
        plt.show()

def plot_results_3d(ds_rec, save_dir):
    for variable in ['true_ext', 'rec_ext', 'error_ext']:
        data = ds_rec[variable].values

        prediction = data
        x_, y_, z_ = prediction.shape
        # print(prediction.shape)

        min = data.min()
        max = data.max()

        X, Y, Z = np.mgrid[0:data.shape[0]:(x_*1j), 0:data.shape[1]:(y_*1j), 0:data.shape[2]:(z_*1j)]
        # print(X.shape, Y.shape, Z.shape)

        # Flatten the coordinates and the data
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=prediction.flatten(),
            isomin=min,
            isomax=max,
            opacity=0.1,
            surface_count=10,
            colorscale='ice',
        ))

        fig.write_html(os.path.join(save_dir, f"predictions_{variable}.html"))

def main(args):
    
    dataset = args.dataset
    scene = args.scene
    data_path = args.data_path
    mode = args.mode
    cloud_id = args.hybridct_cloudid
    vipct_mode = args.vipct_mode

    if dataset not in ['hybridct', 'vipct']:
        raise ValueError("Invalid dataset. Choose either 'hybridct' or 'vipct'.")
    
    save_dir = os.path.join("results", scene) # base save directory
    if not os.path.exists(save_dir): # check if save directory exists
        os.makedirs(save_dir)
    
    if dataset == 'vipct':
        
        print("Using VIPCT dataset for tomographic reconstruction.")

        img_dx = 0.02 # image resolution in [km]
        vol_dx = 0.05 # volume resolution in [km]

        # Select input mode: ML-predicted COT, ground-truth COT
        if mode == "predicted": # use ML-predicted optical thickness field
            tag = "_predCOT"
            print("Tomographic cloud reconstruction using predicted optical thickness field.")
            # load MISR-like optical thickness field from emulator output
            misr = xr.open_dataset(os.path.join(data_path, f"emulator/{scene}.nc")).tau

        elif mode == "truth": # use ground-truth optical thickness field
            tag = "_trueCOT"
            print("Tomographic cloud reconstruction using ground-truth optical thickness field.")
            # load MISR-like optical thickness field from rendering output
            misr = xr.open_dataset(os.path.join(data_path, f"renderer/{scene}.nc")).tau

        # adjust image resolution to match volume resolution
        if vipct_mode == 'coarsen':
            misr = coarsen_image_resolution(misr, res=img_dx, coarse_res=vol_dx)
            print(f"Coarsening image data from {img_dx:.3f} km to {vol_dx:.3f} km resolution.")

        H, W, n_views = misr.shape
        # load VIPCT cloud volume
        scene_path = os.path.join(data_path, f"test/{scene}.pkl")

        with open(scene_path, 'rb') as f:
            data_dict = pickle.load(f)
            # generate cloud dataset from the VIPCT data
            # pad to account for the difference to the images
            total_pad = 15 if H>32 else 0 # pad by 15 pixels to account for size difference between images and volume
            # flipping the x and y axes is needed because when the multiangle_views are generated, the x and y axes are swapped.
            ext_ = generate_cloud_file_dataset_vipct(data_dict, total_pad=total_pad, flip_x=True, flip_y=True, flip_xy=False) 
            # adjust volume resolution to match image resolution
            if vipct_mode == 'upsample':
                ext_ = upsample_volume(ext_, new_res=img_dx)
                print(f"Upsampling volume data from {vol_dx:.3f} to {img_dx:.3f} km resolution.")
            # get volume resolution in [km]
            dx = ext_.delx.data
            dz = ext_.delz.data

        # create 2D cloud mask
        binary_image = get_cloud_mask(ext_.ext, threshold=0.2)

        # extract cloud properties
        cloud_dict = get_cloud_properties(ext_.ext)

        vzas = [extract_viewing_angles(data_dict, i)[0] for i in range(data_dict['images'].shape[0])]
        azimuths = [extract_viewing_angles(data_dict, i)[1] for i in range(data_dict['images'].shape[0])]
        vzas_signed = [extract_viewing_angles(data_dict, i)[2] for i in range(data_dict['images'].shape[0])]

        angles = misr.vza.data # view angles from VIPCT geometry, should be shape (n_views,)

        # extract multiangle views from MISR-like projections
        multiangle_views = create_aligned_views(images=misr.transpose('vza', 'y', 'x'), data_dict=data_dict, cloud_height=cloud_dict['height'], padding_factor=1)

    elif dataset == 'hybridct':

        print("Using HybridCT dataset for tomographic reconstruction.")

        time = float(scene.split("_")[-1].replace("h","")) # extract time from scene name
        scene_path = os.path.join(data_path, f"ground_truth/{scene}.nc")

        # Select input mode: ML-predicted COT, ground-truth COT, or isolated-cloud tests
        if mode == "predicted": # use ML-predicted optical thickness field
            tag = "_predCOT"
            print("Tomographic cloud reconstruction using predicted optical thickness field.")
            misr = xr.open_dataset(os.path.join(data_path, f"predicted_data/MISR_40m_80x80km_{time:.1f}h_optical_thickness_predicted_toa.nc")).tau_pre

        elif mode == "truth": # use ground-truth optical thickness field
            tag = "_trueCOT"
            print("Tomographic cloud reconstruction using ground-truth optical thickness field.")
            misr = xr.open_dataset(os.path.join(data_path, f"ground_truth/MISR_40m_80x80km_{time:.1f}h_optical_thickness_toa.nc")).tau

        with xr.open_dataset(scene_path) as les_: # load the ground truth LES cloud volume
            les = generate_cloud_file_dataset(les_.lwc.transpose("ny", "nx", "nz"), 10, les_.z, les_.dx, pad=0) # generate cloud dataset from the LES data
            lwc = les.lwc
            # get volume resolution in [km]
            dx = les.delx.data 
            dz = les.delz.data 

        min_diameter = 0.500 # minimum diameter in [km]
        min_size = min_diameter / dx # minimum size in pixels

        # compute extinction coefficient (km^-1) from effective radius and wavelength
        beta_ext = get_beta_ext()

        # create 2D cloud mask
        binary_image = get_cloud_mask(lwc, threshold=0.2)

        # Label all connected regions in the binary image
        label_image = measure.label(binary_image) 
        # Calculate properties of the labeled regions
        props_ = measure.regionprops(label_image) 
        # Filter clouds based on minimum size
        label_image, props = filter_labels(binary_image, props_, min_size) 

        print(f"Number of clouds remaining after filtering: {len(props)}/{len(props_)}")

        # select the cloud to analyze
        if cloud_id is None:
            cloud_id = np.random.choice(range(1, len(props)+1)) # randomly select a cloud ID to analyze

        print(f"Analyzing cloud ID: {cloud_id}")

        # extract region and slice for the selected cloud
        region = props[cloud_id]

        save_dir = os.path.join(save_dir, f"cloud_id{cloud_id:03d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # extract cloud properties
        slx, sly = region.slice

        cloud_dict = get_cloud_properties(lwc, slx=slx, sly=sly, beta_ext=beta_ext)
        cloud_COM = cloud_dict['center_of_mass'] - 120 # cloud center of mass height in [km]

        angles = np.sort(misr.vza.data) # view angles from MISR-like geometry

        # extract multiangle views from MISR-like projections
        multiangle_views = proj_utils.generate_views(
            _data = misr,
            slx = slx,
            sly = sly,
            dx = dx,
            cloud_COM = cloud_COM,
            angles = angles,
            offset = [-0.05, -0.04, -0.03, -0.01, 0, 0, 0, 0, 0]
        )

    # get padded cloud data for reconstruction
    test_cloud, test_cloud_padded = get_test_cloud(cloud_dict, dx=dx, dz=dz) # padded test cloud volume, shape (nx_new, ny_new, nz_new)
    test_cloud_3d = test_cloud_padded.data.T # ground truth data for reconstruction, shape (nz_new, ny_new, nx_new)

    # get (padded) sinogram 
    sinogram = multiangle_views[:,:,::-1] # multi_angle_views are shape (nx, ny, n_angles), reverse angle axis
    sinogram_padded, offset_z = get_padded_sinogram(sinogram, test_cloud, test_cloud_padded, angles, dx) # pad sinogram to match reconstruction size

    reconstruction = get_sart_reconstruction(sinogram_padded, test_cloud_padded, angles, dx)
    error = reconstruction - test_cloud_3d # compute reconstruction error

    # yloc = len(test_cloud_padded.y.data) // 2 # select y-location for plotting (middle of the volume)
    yloc = len(test_cloud_padded.y.data) // 2 # select y-location for plotting (middle of the volume)

    # plot reconstruction results
    plot_results(
        test_cloud_3d = test_cloud_3d,
        reconstruction = reconstruction,
        error = error,
        yloc = yloc,
        offset_z = offset_z,
        dx = dx,
        save_dir = save_dir,
        tag = tag
    )

    # save reconstruction volume to netcdf
    ds_rec = xr.Dataset({
        "true_ext": (("x", "y", "z"), test_cloud_3d.T),
        "rec_ext": (("x", "y", "z"), reconstruction.T),
        "error_ext": (("x", "y", "z"), error.T),
        },
        coords={
            "x": test_cloud_padded.x.data, # NOTE: double check whether we should use padded coordinates here
            "y": test_cloud_padded.y.data,
            "z": test_cloud_padded.z.data,
        },
        attrs={
            "cbh_km": cloud_dict['cbh'],
            "cth_km": cloud_dict['cth'],
            "cloud_com_km": cloud_dict['center_of_mass'],
            "cloud_height_km": cloud_dict['height'],
            "unit": "km^-1",
        }
    )
    ds_rec.to_netcdf(os.path.join(save_dir, f"reconstruction_{scene}.nc"))
    print(f"Saved reconstruction results to {os.path.join(save_dir, f'reconstruction_{scene}.nc')}")

    # save 3D html file
    plot_results_3d(ds_rec, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default='hybridct', choices=['hybridct', 'vipct'],
    #                     help="Dataset to use: 'hybridct' or 'vipct'")
    # parser.add_argument("--scene", type=str, default="wc_les_RICO_40m_80kmx80km_T_qc_30.0h", # default='cloud_results_6032'
    #                     help="Name of the LES scene to analyze")
    # parser.add_argument("--data_path", type=str, default="/Users/annajungbluth/Desktop/data/tomography/nasa-jpl/", # default='/Users/annajungbluth/Desktop/data/tomography/technion/VIPCT/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m (incomplete)'
    #                     help="Path to the data directory")
    # parser.add_argument("--mode", choices=["predicted", "truth"],
    #                     default="truth", help="prediction mode")
    # # HybridCT-specific arguments
    # parser.add_argument("--hybridct_cloudid", type=int, default=54,
    #                     help="Cloud ID to analyze in HybridCT dataset")
    # # VIPCT-specific arguments
    # parser.add_argument("--vipct_mode", choices=["coarsen", "upsample", "none"],
    #                     default="none", help="resolution adjustment mode for VIPCT dataset")
    
    parser.add_argument("--dataset", type=str, default='vipct', choices=['hybridct', 'vipct'],
                        help="Dataset to use: 'hybridct' or 'vipct'")
    parser.add_argument("--scene", type=str, default='cloud_results_6032',
                        help="Name of the LES scene to analyze")
    parser.add_argument("--data_path", type=str, default='/Users/annajungbluth/Desktop/data/tomography/technion/VIPCT/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m (incomplete)',
                        help="Path to the data directory")
    parser.add_argument("--mode", choices=["predicted", "truth"],
                        default="truth", help="prediction mode")
    # HybridCT-specific arguments
    parser.add_argument("--hybridct_cloudid", type=int, default=54,
                        help="Cloud ID to analyze in HybridCT dataset")
    # VIPCT-specific arguments
    parser.add_argument("--vipct_mode", choices=["coarsen", "upsample", "none"],
                        default="none", help="resolution adjustment mode for VIPCT dataset")
    
    args = parser.parse_args()
    main(args)

    # TODO:x
    # x Double check the coarsening & padding. I am not convinced that they images align
    # x Creating the sinogram also looks wrong. The clouds move around a lot
    # x Check the padding of the sinogram
    # - Debug implementing the reconstruction and plotting
    # - Check that I didn't break anything for the HybridCT case while implementing the VIPCT case

