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
from datetime import datetime

"""
analyze_mystic_scene.py
Utility script to load LES cloud scenes and perform tomographic
reconstruction experiments using predicted or ground-truth
optical thickness (COT) fields. Produces diagnostics and plots
comparing reconstructed extinction to truth.
"""

def crop_back(cloud, pad=10):
    """
    Crop empty borders from a 2D cloud slice and add padding.

    Returns the cropped array and the slices used for x and z.
    """
    # find non-empty columns
    left = np.where(cloud.sum(axis=0)>0)[0][0] # leftmost non-empty column
    right = np.where(cloud.sum(axis=0)>0)[0][-1] # rightmost non-empty column
    base = np.where(cloud.sum(axis=1)>0)[0][0] # bottommost non-empty row
    top = np.where(cloud.sum(axis=1)>0)[0][-1] # topmost non-empty row
    slz = slice(base-pad, top+pad) # add some padding
    slx = slice(left-pad, right+pad) # add some padding
    return cloud[slz, slx], slz, slx # return cropped cloud and slices

def get_beta_ext(reff=10., wvl=0.67, mie_table="~/libRadtran/data/wc/mie/wc.sol.mie.cdf"):
    """
    Function to get extinction coefficient beta_ext [1/km]
    Args:
        reff (in micron, here homogeneous reff)
        wvl (in micron, set to 0.67 um)
        mie_table: path to mie table NetCDF file
    Returns:
        beta_ext [1/km]
    """
    #reff = np.ones_like(lwc) * reff_value
    for_what = f"for reff={reff}um, wvl={wvl}um" # description
    if os.path.exists(os.path.expanduser(mie_table)): # check that the mie table exists
        optprop = xr.open_dataset(mie_table) # load optical properties
        ilam = np.where(optprop.wavelen==wvl)[0][0] # find wavelength index
        ireff = np.where(optprop.reff==reff)[0][0] # find reff index
        beta_ext = optprop.ext.sel(nlam=ilam, nreff=ireff).data # load corresponding extinction
        print(f"Reading extinction from file {for_what}: {beta_ext} 1/km") 
    elif reff==10. and wvl==0.67:
        beta_ext = 157.71496034205362 # source: libRadtran, load default value if no mie table is provided
        print(f"Setting extinction {for_what}: {beta_ext} 1/km")
    else:
        raise NotImplementedError(f"Provide optical properties {for_what}.")
    return beta_ext

def coarsen_resolution(dataset, res, coarse_res):
    """
    Return coarsened dataset based on original resolution `res`
    to coarse resolution `coarse_res`.
    """
    window_size = int(coarse_res/res) # compute coarsening window size
    print(f"Coarsening dataset using window size {window_size}")
    # perform coarsening of dataset
    # note: boundary='trim' to avoid partial windows
    coarse = dataset.coarsen(x=window_size, boundary='trim').mean().coarsen(y=window_size, boundary='trim').mean().coarsen(z=window_size, boundary='trim').mean()
    return coarse # shape should be (nx//window_size, ny//window_size, nz//window_size)


def filter_labels(binary_image, props, min_size):
    """
    Filter labeled regions by equivalent diameter.

    Removes small regions from `binary_image` in-place and
    returns a relabeled image and updated region properties.
    """
    filtered_props = [] # list to hold filtered properties
    for prop in props: # iterate over region properties
        # prop contains:
        # area, 
        # area_bbox, 
        # area_convex, 
        # area_filled, 
        # axis_major_length, 
        # axis_minor_length, 
        # bbox
        # centroid
        # centroid_local
        # coords
        # eccentricity
        # equivalent_diameter_area
        # euler_number
        # extent
        # feret_diameter_max
        # image
        # image_convex
        # image_filled
        # inertia_tensor
        # inertia_tensor_eigvals
        # label
        # moments
        # moments_central
        # moments_hu
        # moments_normalized
        # orientation
        # perimeter
        # perimeter_crofton
        # slice
        # solidity

        # filter out small clouds based on equivalent diameter
        if prop.equivalent_diameter < min_size: # check size
            binary_image[prop.slice[0], prop.slice[1]] = 0 # remove region with small values
        else:
            filtered_props.append(prop) # keep region
    label_image = measure.label(binary_image) # relabel the filtered binary image
    props = measure.regionprops(label_image) # recalculate region properties
    return label_image, props


def generate_cloud_file_dataset(lwc_, reff_, z, dx, pad=0):
    """
    Create an xarray.Dataset representing the cloud fields.

    Pads the input arrays and builds coordinate arrays for x, y, z.
    """
    nx, ny, nz = lwc_.shape # NOTE: double check order, as input is (ny, nx, nz)

    # add padding
    lwc = np.zeros((nx+2*pad, ny+2*pad, nz)) # padded LWC array
    reff = np.ones((nx+2*pad, ny+2*pad, nz)) * 10 # padded reff array
    slx = slice(pad,nx+pad) # slice for x dimension with padding
    sly = slice(pad,ny+pad) # slice for y dimension with padding
    slz = slice(pad,nz+pad) # slice for z dimension with padding
    lwc[slx,sly,:] = lwc_.data # insert original LWC data
    nx, ny, nz = lwc.shape

    # build coordinate arrays
    coords = {
            "x": np.arange(0, nx*dx, dx),
            "y": np.arange(0, ny*dx, dx),
            "z": z
            }
    x = np.arange(0, nx*dx, dx)[:nx] # define x-coordinates
    y = np.arange(0, ny*dx, dx)[:ny] # define y-coordinates
    z = z[:nz] # define z-coordinates
    ds = xr.Dataset({
        "lwc": xr.DataArray(lwc, dims=["x", "y", "z"], coords=[x, y, z]),
        "reff": xr.DataArray(reff, dims=["x", "y", "z"], coords=[x, y, z]),
        "delx": dx,
        "dely": dx,
        "delz": z[1] - z[0] if len(z) > 1 else 0
        })
    
    return ds # returns padded dataset

def generate_cloud_file_netcdf(filename, lwc_, reff_, z, dx, pad=1):
    """Helper to write the generated cloud dataset to NetCDF."""
    ds = generate_cloud_file_dataset(lwc_, reff_, z, dx, pad) # create padded dataset
    ds.to_netcdf(filename) # save to NetCDF file

def generate_cloud_file(filename, lwc, reff, z, dx):
    # NOTE: Not currently used
    """
    Generate a cloud file in a format compatible with MYSTIC/SHDOM.

    Extracts domain extents and cloud base/top information.
    """
    mystic_domain = {}
    cloud_params = {}
    nx, ny, nz = lwc.shape
    mystic_domain["nx"] = nx 
    mystic_domain["ny"] = ny
    mystic_domain["nz"] = nz
    mystic_domain["dx"] = dx
    mystic_domain["dy"] = dx
    base_idx = np.where(np.sum(lwc, axis=(0,1)) > 0)[0][0] # find cloud base index, sum across x and y, and find first non-zero
    top_idx = np.where(np.sum(lwc, axis=(0,1)) > 0)[0][-1] # find cloud top index, sum across x and y, and find last non-zero
    cloud_params["z_base"] = z[base_idx]
    cloud_params["z_top"] = z[top_idx]
    cloud_params["z"] = z
    cloud_params["lwc"] = lwc
    cloud_params["reff"] = reff
    # generate_cloud_file_shdom(filename, mystic_domain, cloud_params) # TODO: Check what this function is supposed to be replaced wth


def compute_max_view_angle(cth1, cbh1, com1, cth2, cbh2, com2, width2):
    """
    Compute maximum view angle between two cloud columns.

    Returns the angle in degrees from geometry parameters.
    """
    delta_x = com1 - (com2 + width2/2.) # horizontal distance between cloud edges
    delta_z = cth2 - cbh1 # vertical distance between cloud tops and bases
    view_angle = np.rad2deg(np.arctan2(delta_x / delta_z)) # compute angle in degrees
    return view_angle # return the computed angle

def save2netcdf(cloud_lwc, cloud_ext, multi_angle_views, angles, cloud_attrs, nc_fname):
    """Save selected cloud variables and projections into a NetCDF file."""
    res = xr.Dataset(data_vars={"lwc": cloud_lwc, "ext": cloud_ext, "sinogram": multi_angle_views}, attrs=cloud_attrs) # create dataset
    print("save 2 netcdf")
    res.to_netcdf(nc_fname)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predicted", "truth", "isolated"],
                        # default="predicted", help="prediction mode")
                        default="truth", help="prediction mode")
    mode = parser.parse_args().mode

    # Select input mode: ML-predicted COT, ground-truth COT, or isolated-cloud tests
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
    # TODO: Make this more modular to select different scenes
    time = 30. # define time of the LES scene to analyze
    scene = f"wc_les_RICO_40m_80kmx80km_T_qc_{time:.1f}h" # scene name
    _save_dir = os.path.join("results", scene) # base save directory
    if not os.path.exists(_save_dir): # check if save directory exists
        os.makedirs(_save_dir)
    # data_path = "../data"
    data_path = "../data/tomography/nasa-jpl/"
    # Load MISR optical thickness (either truth or ML prediction)
    if mode == "truth":
        misr = xr.open_dataset(os.path.join(data_path, f"ground_truth/MISR_40m_80x80km_{time:.1f}h_optical_thickness_toa.nc")).tau
    else:
        # NOTE: Fixed manual path to predicted COT file to use "time" variable
        misr = xr.open_dataset(os.path.join(data_path, f"predicted_data/MISR_40m_80x80km_{time:.1f}h_optical_thickness_predicted_toa.nc")).tau_pre
    dx_misr = 0.28 # MISR pixel resolution in km
    with xr.open_dataset(os.path.join(data_path, f"ground_truth/{scene}.nc")) as les_: # load the ground truth LES cloud volume
        les = generate_cloud_file_dataset(les_.lwc.transpose("ny", "nx", "nz"), 10, les_.z, les_.dx, pad=0) # generate cloud dataset from the LES data
        lwc = les.lwc
        dx = les.delx.data # resolution in [m]

        # compute extinction coefficient (km^-1) from effective radius and wavelength
        beta_ext = get_beta_ext()

        # create 2D cloud mask by vertically summing LWC and thresholding
        lwc_proj = lwc.sum(axis=2) # this should be shape (nx, ny)
        cloud_mask = lwc_proj > 0.2 # threshold to create binary cloud mask, still in dataset format
        binary_image = cloud_mask.data # convert to numpy array
        # Label the objects in the image
        # Default to connectivity=None, i.e. data.ndim
        label_image_ = measure.label(binary_image) # labels all connected regions with "1"s, i.e. labels each cloud
        # Calculate properties of the labeled regions
        props_ = measure.regionprops(label_image_) # gets properties of each labeled region, for instance, for each cloud, we get the area, bbox, centroid, equivalent_diameter, etc.
        min_diameter = 0.500 # [km]
        min_size = min_diameter / dx # pixels
        print(f"resolution {dx} km")
        # 
        label_image, props = filter_labels(binary_image, props_, min_size) # Filter out small clouds based on equivalent diameter
        print(f"Number of clouds detected: {len(props)}") # 441/6678 clouds remaining
        # iterate over a selected region (example slice) and run reconstruction
        # Select one cloud for testing --> this seems to be a pretty small cloud
        for ir, region in enumerate(props[54:55]):
        # for ir, region in enumerate([props[123], props[88]]):
            t0 = datetime.now()
            save_dir = os.path.join(_save_dir, f"cloud_id{region.label:03d}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            slx, sly = region.slice
            if ir == 54: 
                yloc = 20 # for a specific cloud, set fixed y-location for slice
            else:
                yloc = int(np.ceil((sly.stop - sly.start)/2.)) # center y-location of the cloud region
            # Extract the rectangular region corresponding to the selected cloud from the original image
            cloud_lwc = lwc[slx, sly, :]
            cloud_ext = cloud_lwc * beta_ext # compute extinction field using constant beta_ext and LWC
            cloud_levels = cloud_lwc.z.where(np.sum(cloud_lwc, axis=(0,1))>0).dropna("z").data # get cloud levels in z where there is non-zero LWC
            cbh = cloud_levels.min() # cloud base height
            cth = cloud_levels.max() # cloud top height
            cloud_COM = (cbh + cth) / 2 # cloud center of mass height
            height = cth - cbh # cloud height
            width = region.equivalent_diameter * dx # cloud width in km. region.equivalent_diameter is in pixels, and dx is in km/pixel
            width_x = (slx.stop - slx.start) * dx # cloud width in x-direction
            width_y = (sly.stop - sly.start) * dx # cloud width in y-direction
            aspect_ratio = width / height # cloud aspect ratio
            aspect_ratio_x = width_x / height # cloud aspect ratio in x-direction
            aspect_ratio_y = width_y / height # cloud aspect ratio in y-direction
            angles = np.sort(misr.vza.data) # view angles from MISR-like geometry
            # generate multi-angle projections (sinograms) from MISR-like views
            multi_angle_views = proj_utils.generate_views(misr, slx, sly, dx, cloud_COM=cloud_COM-120, angles=angles, offset=[-0.05,-0.04,-0.03,-0.01,0,0,0,0,0])

            pixel_offset_z = int(len(cloud_ext.z.data)//2 - np.argmin(abs(cloud_ext.z.data - cloud_COM))) # shift cloud in z to center around COM
            _test_cloud = cloud_ext # extract test cloud extinction field
            _test_cloud = _test_cloud.shift(z=pixel_offset_z, fill_value=0) # shift cloud in z to center around COM
            # Pad test cloud to square
            nx, ny, nz = _test_cloud.shape # get shape of test cloud
            nx_new = ny_new = nz_new = max(nx, ny, nz) # new size to pad to (make cube), e.g. (23, 37, 66) -> (66, 66, 66)
            offset_x = int((nx_new-nx)/2) # compute offsets for padding, e.g. (66-23)/2 = 21.5 -> 21
            offset_y = int((ny_new-ny)/2) # compute offsets for padding, e.g. (66-37)/2 = 14.5 -> 14
            offset_z = int((nz_new-nz)/2) # compute offsets for padding, e.g. (66-66)/2 = 0
            pad_x = (offset_x, nx_new-nx-offset_x) # padding for x, e.g. (21, 22)
            pad_y = (offset_y, ny_new-ny-offset_y) # padding for y, e.g. (14, 15)
            pad_z = (offset_z, nz_new-nz-offset_z) # padding for z, e.g. (0, 0)
            test_cloud = _test_cloud.pad({"x": pad_x, "z":pad_z}, constant_values=0) # pad in x and z with zeros, new shape should be (66, 37, 66)
            new_coords = np.concatenate([
                _test_cloud.x.data[0] + np.arange(-pad_x[0], 0) * dx,  # Before padding
                _test_cloud.x.data,                                      # Original coordinates
                _test_cloud.x.data[-1] + np.arange(1, pad_x[1] + 1) * dx  # After padding
            ])
            test_cloud = test_cloud.assign_coords({"x": new_coords})
            rotation_center = test_cloud.data.T.shape[0]//2 # center of rotation for radon transform, e.g. 66//2 = 33
            # Compute reference sinogram from the test cloud using custom radon

            # get sinogram of the test cloud at center y-location from the original cloud data
            # yloc is the center y-location of the cloud region
            # we are not actually using this for anything other than getting the shape
            sino_test = custom_radon(test_cloud.data.T[:,yloc,:], theta=angles, resolution=dx) # radon image of shape (66, 9)

            _sino_proj = multi_angle_views[:,:,::-1] # multi_angle_views are shape (nx, ny, n_angles), reverse angle axis
            # test sinogram
            nx, ny, nz = _sino_proj.shape # nx = 23, ny = 37, nz = 9
            nx_new, nz_new = sino_test.shape # nx_new = 66, nz_new = 9
            offset_x = int((nx_new-nx)/2) # compute offsets for padding, e.g. (66-23)/2 = 21.5 -> 21
            offset_z = int((nz_new-nz)/2) # compute offsets for padding, e.g. (9-9)/2 = 0
            pad_x = (offset_x, nx_new-nx-offset_x) # padding for x, e.g. (21, 22)
            pad_z = (offset_z, nz_new-nz-offset_z) # padding for z, e.g. (0, 0)
            _sino_proj_pad = _sino_proj.pad({"x": pad_x, "z":pad_z}, constant_values=0) # pad in x and z with zeros, new shape should be (66, 37, 9)
            new_coords = np.concatenate([
                _test_cloud.x.data[0] + np.arange(-pad_x[0], 0) * dx,  # Before padding
                _test_cloud.x.data,                                      # Original coordinates
                _test_cloud.x.data[-1] + np.arange(1, pad_x[1] + 1) * dx  # After padding
            ])
            # sino_proj_pad is basically an x-padded version of the different camera views
            sino_proj_pad = _sino_proj_pad.assign_coords({"x": new_coords})

            # Tomographic reconstruction of projected MYSTIC data
            data_test = test_cloud.data.T # shape is (66, 37, 66), from (x, y, z) to (z, y, x)
            data_rec_proj = np.zeros_like(data_test) # array to hold reconstructed data
            niter = 100
            for i in range(sino_proj_pad.shape[1]): # iterate over each projection, i.e. each y-location
                mask = data_test[:,i] == 0 # shape = (66, 66), i.e. (nz, nx), mask of where the true extinction is zero
                prior_proj = np.zeros_like(data_test[:,i]) # initial prior for SART reconstruction, shape = (66, 66) of zeroes
                if mode == "isolated":
                    sino_direct = custom_radon(test_cloud.data.T[:,i,:], theta=angles, resolution=dx)
                for iiter in range(niter):
                    if mode == "isolated":
                        sl_rec = iradon_sart_custom(sino_direct, theta=angles, image=prior_proj, resolution=dx)
                    else:
                        sl_rec = iradon_sart_custom(sino_proj_pad.data[:,i,:], theta=angles, image=prior_proj, resolution=dx)
                    # apply cloud mask
                    sl_rec[mask] = 0 # this mask comes from the ground truth extinction field and would not be available in practice
                    # clip negative results to zero
                    sl_rec[sl_rec<0] = 0
                    # use current estimate as prior
                    prior_proj = sl_rec
                data_rec_proj[:,i,:] = sl_rec

            error_proj = data_rec_proj - data_test

            t1 = datetime.now()
            print(f"Reconstruction of cloud id={region.label} with area {region.area} took {(t1 - t0).total_seconds():.2f} seconds.")

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

            # save data_rec_proj volume to netcdf
            ds_rec = xr.Dataset({
                "true_ext": (("x", "y", "z"), data_test.T),
                "rec_ext": (("x", "y", "z"), data_rec_proj.T),
                "error_ext": (("x", "y", "z"), error_proj.T),
                },
                coords={
                    "x": test_cloud.x.data,
                    "y": _test_cloud.y.data,
                    "z": test_cloud.z.data,
                },
                attrs={
                    "cloud_id": region.label,
                    "area_pixels": region.area,
                    "area_km2": region.area * dx**2,
                    "cbh_km": cbh,
                    "cth_km": cth,
                    "cloud_com_km": cloud_COM,
                    "cloud_height_km": height,
                    "cloud_width_km": width,
                    "aspect_ratio": aspect_ratio,
                    "aspect_ratio_x": aspect_ratio_x,
                    "aspect_ratio_y": aspect_ratio_y,
                    "niter_sart": niter,
                    "unit": "km^-1",
                }
            )
            ds_rec.to_netcdf(os.path.join(save_dir, f"reconstruction_results_cloud_id{region.label:03d}{tag}.nc"))
            print(f"Saved reconstruction results to {os.path.join(save_dir, f'reconstruction_results_cloud_id{region.label:03d}{tag}.nc')}")

