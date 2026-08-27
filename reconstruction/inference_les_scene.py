import os

# stop the BLAS backends from oversubscribing cores when clouds are reconstructed
# in parallel. Must happen before numpy is imported to take effect.
for _threads_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads_var, "1")

import csv
import time
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from skimage import measure
from scipy import stats
from tqdm import tqdm

import projection_utils as proj_utils
from custom_sart import iradon_sart_custom
from analyze_mystic_scene import get_beta_ext, filter_labels, generate_cloud_file_dataset

"""
inference_les_scene.py
Loop over every cloud in a HybridCT LES scene, reconstruct each one
tomographically with SART, and compile the results into

  1. a single 2D netCDF of the whole scene holding the column-integrated
     (z-integrated) optical thickness of the truth, the reconstruction and
     the nadir view, plus a cloud label map, and
  2. a CSV with one row of metrics per cloud.
  3. a parquet file with one row per cloud, all voxels and some associated data

Clouds whose reconstruction fails are left as NaN in the netCDF and flagged
with status="failed" in the CSV.
"""

MISR_ANGLES = np.array([-70.5, -60.0, -45.6, -26.1, 0.0, 26.1, 45.6, 60.0, 70.5]) # MISR view angles in degrees

IDENTITY_COLUMNS = ["cloud_id", "status", "error_msg", "scene", "mode"]
GEOMETRY_COLUMNS = [
    "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1",
    "centroid_x_km", "centroid_y_km", "area_km2", "equiv_diameter_km",
    "n_columns", "n_cloud_voxels",
    "cbh_km", "cth_km", "height_km", "com_km",
]
METRIC_COLUMNS = [
    # 3D extinction, over in-cloud voxels (footprint & true > 0)
    "mean_true_ext", "std_true_ext", "max_true_ext",
    "mean_rec_ext", "std_rec_ext",
    "rmse", "rmse_all", "mae", "mae_std", "bias", "bias_std", "rmse_norm", "r",
    # 2D column-integrated optical thickness, over footprint columns
    "cot_true_mean", "cot_true_std",
    "cot_rec_mean", "cot_rec_std",
    "cot_nadir_mean", "cot_nadir_std",
    "cot_bias", "cot_bias_std", "cot_rmse",
]
BOOKKEEPING_COLUMNS = ["niter", "runtime_s"]
CSV_COLUMNS = IDENTITY_COLUMNS + GEOMETRY_COLUMNS + METRIC_COLUMNS + BOOKKEEPING_COLUMNS

class CloudTooLargeError(RuntimeError):
    """
    Raised when a cloud's bounding box exceeds --max_width.

    SART runs once per y-slice on a cube of side max(nx, ny, nz). 
    If the cloud is too big, this will take quite long.
    """

class NaNSinogramError(RuntimeError):
    """
    Raised when the projected views contain NaN.

    generate_views interpolates each off-nadir view onto the nadir window and
    returns NaN wherever that window reaches past the projected data. 
    This seems to be a bug in the current reprojection code.
    """

def resolve_data_file(base_path, subdir, filename):
    """
    Locate the relevant data file, either in the subdirectory or directly in the base path.

    Parameters:
    base_path : str
        Directory holding the data
    subdir : str
        Subdirectory the file is conventionally stored in, e.g. "ground_truth"
    filename : str
        Name of the file to find

    Returns:
    path : str
        Path to the existing file
    """
    for candidate in (os.path.join(base_path, subdir, filename), os.path.join(base_path, filename)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in {base_path} (with or without '{subdir}/')")

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

def get_test_cloud(cloud_dict, dx, dz, shift=True):
    """
    Function to extract and pad the test cloud volume from cloud dictionary
    Parameters:
    cloud_dict : dict
        Dictionary containing cloud properties including 'cloud_ext' DataArray
    dx : float
        Spatial resolution in x-direction [km]
    dz : float
        Spatial resolution in z-direction [km]
    shift : bool
        Whether to shift the cloud in z to center around the center of mass before padding

    Returns:
    test_cloud : xarray.DataArray
        Cloud extinction data array before padding
    test_cloud_padded : xarray.DataArray
        Padded 3D cloud extinction data array
    pad_x : tuple
        Cells padded onto the start and end of x, needed to trim the
        reconstruction back onto the scene grid
    """
    test_cloud = cloud_dict['cloud_ext']
    if shift: # NOTE: This only shifts the cloud and does not correct the z-coordinates
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

    return test_cloud, test_cloud_padded, pad_x # shape (nx_new, ny_new, nz_new)

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

def get_sart_reconstruction(sinogram_padded, test_cloud_padded, angles, dx, niter=100, progress=True):
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
    progress : bool
        Whether to show a progress bar over y-slices. Disabled when looping over
        many clouds, where the outer progress bar is the useful one.

    Returns:
    reconstruction : numpy.ndarray
        Reconstructed 3D cloud extinction data array
    """
    test_cloud_data = test_cloud_padded.data.T # from (x, y, z) to (z, y, x) for reconstruction, shape (nz_new, ny_new, nx_new)

    reconstruction = np.zeros_like(test_cloud_data) # array to hold reconstructed data

    for y in tqdm(range(sinogram_padded.shape[1]), disable=not progress): # iterate over each projection, i.e. each y-location

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

def integrate_z(ext, dz):
    """
    Integrate an extinction volume along z to get optical thickness.

    The LES z-grid is uniform, so this is a plain sum weighted by dz.

    Parameters:
    ext : numpy.ndarray
        Extinction volume in [km^-1], shape (nz, ny, nx)
    dz : float
        Vertical grid spacing in [km]

    Returns:
    numpy.ndarray
        Column-integrated optical thickness (dimensionless), shape (nx, ny)
    """
    return ext.sum(axis=0).T * dz

def get_mean_std(values):
    """Return the (mean, std) of an array, or (nan, nan) if it is empty."""
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))

def get_pearson_r(x, y):
    """
    Pearson correlation that degrades to NaN instead of raising.

    Single-column clouds produce constant arrays, for which the correlation is
    undefined; those clouds still get a valid row with r = NaN.
    """
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(stats.pearsonr(x, y)[0])

def compute_metrics(true_ext, rec_ext, cot_nadir, footprint, dz):
    """
    Compute reconstruction metrics for a single cloud over its footprint.

    Parameters:
    true_ext : numpy.ndarray
        True extinction trimmed to the bounding box, shape (nz, ny, nx)
    rec_ext : numpy.ndarray
        Reconstructed extinction trimmed to the bounding box, shape (nz, ny, nx)
    cot_nadir : numpy.ndarray
        Nadir-view optical thickness over the bounding box, shape (nx, ny)
    footprint : numpy.ndarray
        Boolean mask of the cloud within its bounding box, shape (nx, ny)
    dz : float
        Vertical grid spacing in [km]

    Returns:
    metrics : dict
        Metric name to value
    cot_rec : numpy.ndarray
        Column-integrated reconstruction over the bounding box, shape (nx, ny)
    voxels : tuple of numpy.ndarray
        The true and reconstructed extinction of every in-cloud voxel, kept so
        that per-voxel scatter and error plots can be made after the fact
    """
    metrics = {}

    # the footprint is 2D, so broadcast it over z to select the cloud's columns
    footprint_3d = footprint.T[None, :, :] # (nx, ny) -> (1, ny, nx)
    in_box = np.broadcast_to(footprint_3d, true_ext.shape)
    in_cloud = in_box & (true_ext > 0)

    true_vals = true_ext[in_cloud]
    rec_vals = rec_ext[in_cloud]
    error = rec_vals - true_vals

    metrics["n_cloud_voxels"] = int(in_cloud.sum())
    metrics["mean_true_ext"], metrics["std_true_ext"] = get_mean_std(true_vals)
    metrics["max_true_ext"] = float(true_vals.max()) if true_vals.size else np.nan
    metrics["mean_rec_ext"], metrics["std_rec_ext"] = get_mean_std(rec_vals)

    metrics["rmse"] = float(np.sqrt(np.mean(error**2))) if error.size else np.nan
    metrics["mae"], metrics["mae_std"] = get_mean_std(np.abs(error))
    metrics["bias"], metrics["bias_std"] = get_mean_std(error)
    metrics["rmse_norm"] = (
        metrics["rmse"] / metrics["mean_true_ext"] if metrics["mean_true_ext"] else np.nan
    )
    metrics["r"] = get_pearson_r(true_vals, rec_vals)

    # the same error over every voxel in the footprint columns, zeros included
    error_all = rec_ext[in_box] - true_ext[in_box]
    metrics["rmse_all"] = float(np.sqrt(np.mean(error_all**2))) if error_all.size else np.nan

    # 2D column-integrated metrics, over the footprint columns
    cot_true = integrate_z(true_ext, dz)
    cot_rec = integrate_z(rec_ext, dz)

    cot_true_vals = cot_true[footprint]
    cot_rec_vals = cot_rec[footprint]
    cot_nadir_vals = cot_nadir[footprint]
    cot_error = cot_rec_vals - cot_true_vals

    metrics["cot_true_mean"], metrics["cot_true_std"] = get_mean_std(cot_true_vals)
    metrics["cot_rec_mean"], metrics["cot_rec_std"] = get_mean_std(cot_rec_vals)
    metrics["cot_nadir_mean"], metrics["cot_nadir_std"] = get_mean_std(cot_nadir_vals)
    metrics["cot_bias"], metrics["cot_bias_std"] = get_mean_std(cot_error)
    metrics["cot_rmse"] = float(np.sqrt(np.mean(cot_error**2))) if cot_error.size else np.nan
    return metrics, cot_rec, (true_vals.astype("float32"), rec_vals.astype("float32"))

def get_cloud_geometry(prop, dx):
    """
    Extract the geometric descriptors of a labelled cloud region.

    Parameters:
    prop : skimage.measure.RegionProperties
        Region properties of a single labelled cloud
    dx : float
        Horizontal grid resolution in [km]

    Returns:
    dict
        Bounding box, centroid, area and equivalent diameter of the cloud
    """
    slx, sly = prop.slice
    return {
        "bbox_x0": slx.start,
        "bbox_x1": slx.stop,
        "bbox_y0": sly.start,
        "bbox_y1": sly.stop,
        "centroid_x_km": float(prop.centroid[0] * dx),
        "centroid_y_km": float(prop.centroid[1] * dx),
        "area_km2": float(prop.area * dx * dx),
        # computed directly rather than read off prop, whose attribute name for
        # this quantity has changed across skimage versions
        "equiv_diameter_km": float(np.sqrt(4.0 * prop.area / np.pi) * dx),
        "n_columns": int(prop.area),
    }

def reconstruct_cloud(prop, lwc, misr, cot_nadir_scene, beta_ext, angles, dx, dz,
                      niter, com_offset, view_offset, shift=True, max_width=None):
    """
    Reconstruct a single cloud and score it against the truth.

    Parameters:
    prop : skimage.measure.RegionProperties
        Region properties of the cloud to reconstruct
    lwc : xarray.DataArray
        Scene liquid water content, shape (nx, ny, nz)
    misr : xarray.DataArray
        Multi-angle optical thickness views, shape (nx, ny, n_angles)
    cot_nadir_scene : numpy.ndarray
        Nadir-view optical thickness over the whole scene, shape (nx, ny)
    beta_ext : float
        Extinction coefficient in [km^-1]
    angles : numpy.ndarray
        Sorted view angles in degrees
    dx : float
        Horizontal grid resolution in [km]
    dz : float
        Vertical grid resolution in [km]
    niter : int
        Number of SART iterations
    com_offset : float
        Offset added to the cloud center-of-mass before the parallax correction
    view_offset : list
        Per-view offsets passed to the view generation
    shift : bool
        Whether to center the cloud on its center of mass in z
    max_width : float or None
        Skip the cloud if either side of its bounding box exceeds this width
        in [km]. None reconstructs every cloud regardless of size.

    Returns:
    cot_rec : numpy.ndarray
        Column-integrated reconstruction over the bounding box, shape (nx, ny)
    metrics : dict
        Per-cloud metrics, including cloud base and top height
    """
    slx, sly = prop.slice
    nx = slx.stop - slx.start
    ny = sly.stop - sly.start

    # checked before any work is done, so a skipped cloud costs nothing
    if max_width is not None and max(nx, ny) * dx > max_width:
        raise CloudTooLargeError(
            f"bounding box {nx}x{ny} px ({max(nx, ny)*dx:.1f} km) "
            f"exceeds max_width {max_width:.1f} km")

    cloud_dict = get_cloud_properties(lwc, slx=slx, sly=sly, beta_ext=beta_ext)
    cloud_COM = cloud_dict['center_of_mass'] + com_offset # cloud center of mass height in [km]

    # get padded cloud data for reconstruction
    test_cloud, test_cloud_padded, pad_x = get_test_cloud(cloud_dict, dx=dx, dz=dz, shift=shift) # padded test cloud volume, shape (nx_new, ny_new, nz_new)
    test_cloud_3d = test_cloud_padded.data.T # ground truth data for reconstruction, shape (nz_new, ny_new, nx_new)

    # the z-shift drops anything pushed off the edge of the array, which would
    # silently corrupt the column integral
    mass_before = float(cloud_dict['cloud_ext'].sum())
    mass_after = float(test_cloud_padded.sum())
    if not np.isclose(mass_before, mass_after, rtol=1e-6):
        print(f"  warning: cloud {prop.label} lost {100*(1-mass_after/mass_before):.2f}% of its "
              f"extinction in the center-of-mass shift")

    # extract multiangle views from MISR-like projections
    multiangle_views = proj_utils.generate_views(
        _data = misr,
        slx = slx,
        sly = sly,
        dx = dx,
        cloud_COM = cloud_COM,
        angles = angles,
        offset = view_offset
    )
    sinogram = multiangle_views[:,:,::-1] # multi_angle_views are shape (nx, ny, n_angles), reverse angle axis

    # get (padded) sinogram
    sinogram_padded, _ = get_padded_sinogram(sinogram, test_cloud, test_cloud_padded, angles, dx) # pad sinogram to match reconstruction size

    # raise error if there is a nan in the sinogram, which would cause SART to fail
    n_nan = int(np.isnan(sinogram_padded.data).sum())
    if n_nan:
        raise NaNSinogramError(
            f"{n_nan} NaN values in the projected views "
            f"({100*n_nan/sinogram_padded.data.size:.1f}% of the sinogram)")

    reconstruction = get_sart_reconstruction(
        sinogram_padded, test_cloud_padded, angles, dx, niter=niter, progress=False)

    # trim the x-padding so both volumes line back up with the bounding box
    x_slice = slice(pad_x[0], pad_x[0] + nx)
    true_ext = test_cloud_3d[:, :, x_slice]
    rec_ext = reconstruction[:, :, x_slice]

    metrics, cot_rec, voxels = compute_metrics(
        true_ext = true_ext,
        rec_ext = rec_ext,
        cot_nadir = cot_nadir_scene[slx, sly],
        footprint = prop.image,
        dz = dz,
    )
    metrics.update({
        "cbh_km": float(cloud_dict['cbh']),
        "cth_km": float(cloud_dict['cth']),
        "height_km": float(cloud_dict['height']),
        "com_km": float(cloud_dict['center_of_mass']),
    })

    return cot_rec, metrics, voxels

# Some code for multiprocessing to share the scene data with each worker.
WORKER_STATE = {}

def process_cloud(index):
    """
    Reconstruct one cloud from the shared worker state and build its CSV row.

    Written to be called either directly (serial) or through a process pool, so
    it takes only an index and returns only small objects.

    Parameters:
    index : int
        Position of the cloud in WORKER_STATE["props"]

    Returns:
    index : int
        The index that was passed in, so the caller can match up results that
        arrive out of order
    row : dict
        The cloud's CSV row
    cot_rec : numpy.ndarray or None
        Column-integrated reconstruction over the bounding box, or None if the
        cloud was not reconstructed
    voxels : tuple of numpy.ndarray or None
        True and reconstructed extinction for every in-cloud voxel. Only the two
        arrays are returned: the per-cloud properties that go alongside them in
        the parquet are already in `row`, so repeating them here would just make
        the payload bigger.
    """
    state = WORKER_STATE
    prop = state["props"][index]

    row = {column: np.nan for column in CSV_COLUMNS}
    row.update({
        "cloud_id": prop.label,
        "status": "ok",
        "error_msg": "",
        "scene": state["scene"],
        "mode": state["mode"],
        "niter": state["niter"],
    })
    row.update(get_cloud_geometry(prop, state["dx"]))

    cot_rec = None
    voxels = None
    t_start = time.time()
    try:
        cot_rec, metrics, voxels = reconstruct_cloud(
            prop = prop,
            lwc = state["lwc"],
            misr = state["misr"],
            cot_nadir_scene = state["cot_nadir"],
            beta_ext = state["beta_ext"],
            angles = state["angles"],
            dx = state["dx"],
            dz = state["dz"],
            niter = state["niter"],
            com_offset = state["com_offset"],
            view_offset = state["view_offset"],
            shift = state["shift"],
            max_width = state["max_width"],
        )
        row.update(metrics)
    except CloudTooLargeError as e:
        # deliberately skipped, not a failure: one mesoscale cluster can cost
        # more than the rest of the scene combined
        row["status"] = "too_large"
        row["error_msg"] = str(e)
    except NaNSinogramError as e:
        # not a crash: the views themselves are unusable, so the cloud is
        # reported as unreconstructed rather than scored on NaN
        row["status"] = "nan_sinogram"
        row["error_msg"] = str(e)
    except Exception as e:
        row["status"] = "failed"
        row["error_msg"] = f"{type(e).__name__}: {e}"
    row["runtime_s"] = round(time.time() - t_start, 3)

    return index, row, cot_rec, voxels

def get_estimated_cost(prop, nz):
    """
    Rough relative cost of reconstructing a cloud, used to order the work.

    SART runs once per y-slice on an (nx_new x nz_new) cube, and the measured
    cost per slice grows as roughly cube^2.4, so the largest cloud in a scene
    can cost several hundred times the smallest. Dispatching the expensive ones
    first keeps a parallel run from ending on a single straggler.

    Parameters:
    prop : skimage.measure.RegionProperties
        Region properties of the cloud
    nz : int
        Number of vertical levels, which sets the minimum cube size

    Returns:
    float
        Relative cost estimate, in arbitrary units
    """
    slx, sly = prop.slice
    nx = slx.stop - slx.start
    ny = sly.stop - sly.start
    return ny * max(nx, ny, nz) ** 2.4

def build_voxel_chunk(voxels, row, dx):
    """
    Build one cloud's block of the per-voxel table.

    Parameters:
    voxels : tuple of numpy.ndarray
        True and reconstructed extinction for each in-cloud voxel
    row : dict
        The cloud's CSV row, which already holds its geometry
    dx : float
        Horizontal grid resolution in [km]

    Returns:
    pandas.DataFrame
        One row per in-cloud voxel. The per-cloud properties are given as
        scalars and broadcast down the block by pandas.
    """
    true_vals, rec_vals = voxels
    return pd.DataFrame({
        "cloud_id": row["cloud_id"],
        "true_ext": true_vals,
        "rec_ext": rec_vals,
        # cloud geometry, so error can be conditioned on cloud shape and altitude
        "height_km": row["height_km"],
        "com_km": row["com_km"],
        "width_x_km": (row["bbox_x1"] - row["bbox_x0"]) * dx,
        "width_y_km": (row["bbox_y1"] - row["bbox_y0"]) * dx,
    })

VOXEL_DTYPES = {
    "cloud_id": "int32",
    "true_ext": "float32", "rec_ext": "float32",
    "height_km": "float32", "com_km": "float32",
    "width_x_km": "float32", "width_y_km": "float32",
}

def write_voxel_parquet(voxel_chunks, parquet_path):
    """
    Concatenate the per-cloud voxel blocks and write them to parquet.

    Parameters:
    voxel_chunks : list of pandas.DataFrame
        One block per cloud
    parquet_path : str
        Destination file

    Returns:
    int
        Number of voxels written
    """
    if not voxel_chunks:
        return 0
    # broadcast scalars come out as float64/int64, so set the widths once here
    table = pd.concat(voxel_chunks, ignore_index=True).astype(VOXEL_DTYPES)
    table.to_parquet(parquet_path, index=False, compression="snappy")
    return len(table)

def write_scene_netcdf(fields, coords, attrs, nc_path):
    """
    Write the column-integrated scene to netCDF.

    Called periodically as well as at the end, so that a run killed at its wall
    clock limit still leaves usable output on disk.

    Parameters:
    fields : dict
        Variable name to 2D array on the (x, y) scene grid
    coords : dict
        The x and y coordinates in [km]
    attrs : dict
        Global attributes
    nc_path : str
        Destination file
    """
    ds_scene = xr.Dataset(
        {name: (("x", "y"), values) for name, values in fields.items()},
        coords=coords,
        attrs=attrs,
    )
    encoding = {variable: {"zlib": True, "complevel": 4} for variable in ds_scene.data_vars}
    ds_scene.to_netcdf(nc_path, encoding=encoding)

def sort_csv_by_cloud_id(csv_path):
    """
    Rewrite the metrics CSV in cloud_id order.

    Rows are appended as clouds finish so that an interrupted run keeps its
    results, which leaves them unordered after a parallel run.
    """
    with open(csv_path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row["cloud_id"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def main(args):

    scene = args.scene
    data_path = args.data_path
    volume_path = args.volume_path
    mode = args.mode

    save_dir = os.path.join(args.output_dir, scene) # base save directory
    if not os.path.exists(save_dir): # check if save directory exists
        os.makedirs(save_dir)

    print("Using HybridCT dataset for tomographic reconstruction.")

    time_h = float(scene.split("_")[-1].replace("h","")) # extract time from scene name

    shift = True # if True, shifts the volume to center of mass
    # in the creation of the sinograms, the images are also corrected
    # for the volumes, the shift is in z
    # for the images, the shift is in x

    # Select input mode: ML-predicted COT or ground-truth COT
    if mode == "predicted": # use ML-predicted optical thickness field
        tag = "_predCOT"
        print("Tomographic cloud reconstruction using predicted optical thickness field.")
        misr = xr.open_dataset(resolve_data_file(
            data_path, "predicted_data",
            f"MISR_40m_80x80km_{time_h:.1f}h_optical_thickness_predicted_toa.nc")).tau_pre

    elif mode == "truth": # use ground-truth optical thickness field
        tag = "_trueCOT"
        print("Tomographic cloud reconstruction using ground-truth optical thickness field.")
        misr = xr.open_dataset(resolve_data_file(
            data_path, "ground_truth",
            f"MISR_40m_80x80km_{time_h:.1f}h_optical_thickness_toa.nc")).tau

    # hold the views in memory. generate_views rebuilds a full-scene projection
    # for every cloud, so leaving this lazy re-reads ~300 MB from disk per cloud
    # and dominates the runtime once the LES volume fills the page cache.
    misr = misr.load()

    # load HybridCT cloud volume
    scene_path = resolve_data_file(volume_path, "ground_truth", f"{scene}.nc")
    with xr.open_dataset(scene_path) as les_: # load the ground truth LES cloud volume
        les = generate_cloud_file_dataset(les_.lwc.transpose("ny", "nx", "nz"), 10, les_.z, les_.dx, pad=0) # generate cloud dataset from the LES data
        lwc = les.lwc
        # get volume resolution in [km]
        dx = les.delx.data
        dz = les.delz.data

    # compute extinction coefficient (km^-1) from effective radius and wavelength
    beta_ext = get_beta_ext()

    # create 2D cloud mask
    binary_image = get_cloud_mask(lwc, threshold=0.2)

    # Label all connected regions in the binary image
    label_image = measure.label(binary_image)
    # Calculate properties of the labeled regions
    props = measure.regionprops(label_image)

    if args.min_diameter > 0:
        # Filter clouds based on minimum size
        min_size = args.min_diameter / dx # minimum size in pixels
        n_unfiltered = len(props)
        label_image, props = filter_labels(binary_image, props, min_size)
        print(f"Number of clouds remaining after filtering: {len(props)}/{n_unfiltered}")
        # Update the mask to the filtered clouds
        binary_image = label_image > 0

    # optionally restrict to a subset of clouds, for debugging or smoke tests
    if args.cloud_ids is not None:
        wanted = set(args.cloud_ids)
        props = [prop for prop in props if prop.label in wanted]
    if args.max_clouds is not None:
        props = np.random.choice(props, size=args.max_clouds, replace=False).tolist()

    print(f"Reconstructing {len(props)} clouds.")

    angles = np.sort(misr.vza.data) # view angles from MISR-like geometry

    # scene-wide fields, computed once. Summing before scaling avoids
    # materialising a second copy of the full volume.
    cot_true = (lwc.sum("z") * beta_ext * dz).data.astype("float32")
    cot_nadir = misr.isel(vza=np.abs(misr.vza).argmin()).data.astype("float32")

    # NaN everywhere there is cloud, so that a footprint left unwritten reads as
    # "no reconstruction" rather than as zero optical thickness. Clear sky is a true zero.
    cot_rec = np.full(binary_image.shape, np.nan, dtype="float32")
    cot_rec[~binary_image] = 0.0
    cloud_id_map = np.zeros(binary_image.shape, dtype="int32")

    csv_path = os.path.join(save_dir, f"cloud_metrics{tag}.csv")
    nc_path = os.path.join(save_dir, f"reconstructed_scene_cot{tag}.nc")
    parquet_path = os.path.join(save_dir, f"cloud_voxels{tag}.parquet")

    # write the header up front, then append a row per cloud, so that a run
    # which dies partway through keeps everything already completed
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    # share the scene inputs with the workers before the pool is forked
    WORKER_STATE.update({
        "props": props,
        "lwc": lwc,
        "misr": misr,
        "cot_nadir": cot_nadir,
        "beta_ext": beta_ext,
        "angles": angles,
        "dx": dx,
        "dz": dz,
        "niter": args.niter,
        "max_width": args.max_width,
        "com_offset": args.com_offset,
        "view_offset": args.view_offset,
        "shift": shift,
        "scene": scene,
        "mode": mode,
    })

    # most expensive clouds first, so a parallel run does not finish on a straggler
    order = sorted(range(len(props)),
                   key=lambda i: -get_estimated_cost(props[i], len(lwc.z)))

    pool = None
    if args.n_jobs == 1:
        results = (process_cloud(i) for i in order)
    else:
        # fork so that the workers inherit the scene arrays instead of pickling them
        pool = multiprocessing.get_context("fork").Pool(args.n_jobs)
        results = pool.imap_unordered(process_cloud, order)
        print(f"Reconstructing across {args.n_jobs} processes.")

    counts = {"ok": 0, "failed": 0, "nan_sinogram": 0, "too_large": 0}
    voxel_chunks = []

    scene_fields = lambda: {
        "cot_true": cot_true,
        "cot_rec": cot_rec,
        "cot_mask": binary_image.astype("int8"),
        "cot_nadir": cot_nadir,
        "cloud_id": cloud_id_map,
    }
    scene_coords = {"x": lwc.x.data, "y": lwc.y.data}
    scene_attrs = lambda: {
        "scene": scene,
        "mode": mode,
        "description": "Column-integrated (z-integrated) optical thickness of the LES "
                       "scene. cot_rec is NaN where a cloud's reconstruction failed.",
        "beta_ext_km-1": beta_ext,
        "dx_km": float(dx),
        "dz_km": float(dz),
        "niter": args.niter,
        "n_clouds": len(props),
        "n_done": sum(counts.values()),
        "n_failed": counts["failed"],
        "n_nan_sinogram": counts["nan_sinogram"],
        "n_too_large": counts["too_large"],
        "unit": "dimensionless (optical thickness)",
    }

    def flush():
        """Write the scene and voxel outputs as they stand."""
        write_scene_netcdf(scene_fields(), scene_coords, scene_attrs(), nc_path)
        write_voxel_parquet(voxel_chunks, parquet_path)

    n_done = 0
    try:
        for index, row, cloud_cot_rec, voxels in tqdm(results, total=len(order), desc="clouds"):
            prop = props[index]
            slx, sly = prop.slice
            if cloud_cot_rec is not None:
                # fill only the cloud's own footprint, so that overlapping bounding
                # boxes cannot overwrite each other
                cot_rec[slx, sly][prop.image] = cloud_cot_rec[prop.image]
            cloud_id_map[slx, sly][prop.image] = prop.label
            counts[row["status"]] += 1

            if voxels is not None and voxels[0].size:
                voxel_chunks.append(build_voxel_chunk(voxels, row, dx))

            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

            # checkpoint, so that a run killed at its wall clock limit still
            # leaves the scene and the voxels it has finished on disk
            n_done += 1
            if args.checkpoint_every and n_done % args.checkpoint_every == 0:
                flush()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    sort_csv_by_cloud_id(csv_path)
    flush()

    print(f"Reconstructed {counts['ok']}/{len(props)} clouds ({counts['failed']} failed, "
          f"{counts['nan_sinogram']} with NaN in the projected views, "
          f"{counts['too_large']} skipped as too large).")
    print(f"Saved per-cloud metrics to {csv_path}")
    n_voxels = sum(len(chunk) for chunk in voxel_chunks)
    if n_voxels:
        print(f"Saved {n_voxels:,} in-cloud voxels to {parquet_path}")
    print(f"Saved column-integrated scene to {nc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="wc_les_RICO_40m_80kmx80km_T_qc_30.0h",
                        help="Name of the LES scene to analyze")
    parser.add_argument("--data_path", type=str, default="/home/users/annaju/data/nasa-jpl/30.0h",
                        help="Path to the data directory")
    parser.add_argument("--volume_path", type=str, default="/home/users/annaju/data/nasa-jpl/30.0h",
                        help="Path to the volume data directory")
    parser.add_argument("--output_dir", type=str, default="/home/users/annaju/results",
                        help="Directory to write the scene netCDF and metrics CSV to")
    parser.add_argument("--mode", choices=["predicted", "truth"],
                        default="truth", help="prediction mode")
    parser.add_argument("--niter", type=int, default=100,
                        help="Number of SART iterations per y-slice")
    parser.add_argument("--min_diameter", type=float, default=0.0,
                        help="Minimum cloud equivalent diameter in [km]. 0 keeps every cloud.")
    parser.add_argument("--max_clouds", type=int, default=None,
                        help="Only reconstruct the first N clouds, for smoke tests")
    parser.add_argument("--max_width", type=float, default=None,
                        help="Skip clouds whose bounding box is wider than this in [km], "
                             "recording them as status='too_large'. Cost grows as roughly "
                             "bbox^2.4, so one mesoscale cluster can outweigh a whole scene.")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Rewrite the scene netCDF and voxel parquet every N clouds, so "
                             "that a run killed at its wall clock limit keeps its results. "
                             "0 disables checkpointing.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of worker processes to reconstruct clouds with. "
                             "1 runs everything in the parent process.")
    parser.add_argument("--cloud_ids", type=int, nargs="+", default=None,
                        help="Only reconstruct these cloud labels")
    parser.add_argument("--com_offset", type=float, default=-120.0,
                        help="Offset added to the cloud center-of-mass height before the "
                             "parallax correction. Preserved from the original script.")
    parser.add_argument("--view_offset", type=float, nargs="+",
                        default=[-0.05, -0.04, -0.03, -0.01, 0, 0, 0, 0, 0],
                        help="Per-view offsets applied when generating the multi-angle views")

    args = parser.parse_args()
    main(args)
