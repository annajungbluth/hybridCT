import numpy as np
import xarray as xr
import matplotlib.pylab as plt

def coarsen_resolution(dataset, res, coarse_res):
    """Return a coarsened dataset based on the original resolution `res` and
    the target coarse resolution `coarse_res`.

    Args:
        dataset (xarray.DataArray or xarray.Dataset): Input dataset.
        res (float): Original resolution of the dataset.
        coarse_res (float): Target coarse resolution.

    Returns:
        xarray.DataArray or xarray.Dataset: Coarsened dataset.
    """
    window_size = int(coarse_res/res)
    print(f"Coarsening dataset using window size {window_size}")
    coarse = dataset.coarsen(x=window_size, boundary='trim').mean().coarsen(y=window_size, boundary='trim').mean()
    return coarse

def project2com_repeat(data, delta_z, dx):
    """ project 2 com by repeating data array
    """
    x = data.x.data
    y = data.y.data
    vza = data.vza.data
    nx = len(x)
    views = []
    for _vza in vza:
        tan_factor = np.tan(np.deg2rad(_vza))
        shift = tan_factor * delta_z
        repeat = int(abs(shift)//x.max())
        n_repeat = int(3 + repeat)
        x_shift = x + shift
        print(_vza, shift, n_repeat)
        arrays = []
        for i in range(n_repeat):
            arrays.append(data.sel(vza=_vza).data)
        array_concat = np.concat(arrays, axis=0)
        x_ext = np.linspace(0, x.max()*n_repeat, array_concat.shape[0])
        extended = xr.DataArray(array_concat, dims=["x", "y"], coords={"x": x_ext, "y": y})
        #if shift < 0:
        #    od_proj = extended.interp(x=(repeat*x.max()+dx + x_shift))
        #else:
        od_proj = extended.interp(x=(repeat*x.max() + x_shift))
        od_proj = od_proj.assign_coords({"x": x})
        views.append(od_proj)
    views = xr.concat(views, dim=xr.DataArray(vza, dims="vza", name="vza"))
    return views

def project2com(da, cloud_height_avg, dx):
    """Project DataArray to the cloud center-of-mass.

    Args:
        da (xarray.DataArray): Input data to project.
        cloud_height_avg (float): Projection height.
        dx (float): Grid resolution.

    Returns:
        xarray.DataArray: Projected data.
    """
    data_proj = np.zeros_like(da.data) # create empty array for projected data
    x_coord = da.sel(vza=0).x.data # x-coordinates from nadir view
    for ivza, vza in enumerate(da.vza.data):
        view = da.sel(vza=vza) # select view at given VZA
        # perform parallex correction by projecting to cloud COM height
        idx, grid_new = get_projection_sortIndex(vza, cloud_height_avg, x_coord, dx) # get sort index for projection
        proj_shift = grid_new.min() # calculate projection shift
        _data_proj = xr.DataArray(view.data[idx,:],
                                  coords={"x": view.x.data+proj_shift, "y": view.y.data},
                                  dims=(["x", "y"]))
        data_proj[:,:,ivza] = _data_proj.interp(x=x_coord).data
    # create a copy of da but use projected data
    da_proj = da.copy(data=data_proj)
    return da_proj

def get_closest_index(value, x, dx=0.02):
    """Find the closest index in array `x` for a given value,
    within a specified tolerance.

    Args:
        value (float): The x-value to find the closest index for.
        x (numpy.ndarray or list): Array of values to search within.
        dx (float, optional): Tolerance value to define closeness. Defaults to 0.02.

    Returns:
        tuple: Tuple containing the closest index and the corresponding value in `x`.
    """
    diff = np.abs(value - x)
    # Find indices within the tolerance
    idx = np.where(diff <= dx)[0]
    # Select the index with the minimum difference
    closest_idx = idx[np.argmin(diff[idx])]
    # Access the value corresponding to the selected index
    selected_value = x[closest_idx]
    return closest_idx, selected_value

def wrap_around(dim_slice, data, dx=0.02, dim="x"):
    """Wrap around the dataset along a specified dimension.

    Args:
        dim_slice (slice): The range of values to slice.
        data (xarray.DataArray or xarray.Dataset): The dataset to wrap around.
        dx (float, optional): The grid spacing. Defaults to 0.02.
        dim (str, optional): The dimension to wrap along. Defaults to "x".

    Returns:
        xarray.DataArray or xarray.Dataset: The wrapped dataset.
    """
    dim_max = data[dim].data.max()
    dim_min = data[dim].data.min()
    start = dim_slice.start
    stop = dim_slice.stop
    if start < dim_min:
        idx, ival = get_closest_index(dim_min - start, data[dim].data, dx)
        # move columns to beginning of field in negative x-direction
        # NOTE: shouldn't the dimension be modular, rather than always x?
        return data.roll(x=idx, roll_coords=False).sel(**{dim: slice(dim_min, stop-start)})
    elif stop > dim_max:
        idx, ival = get_closest_index(dim_max-start, data[dim].data, dx)
        # move columns to beginning of field in negative x-direction
        # NOTE: shouldn't the dimension be modular, rather than always x?
        return data.roll(x=idx, roll_coords=False).sel(**{dim: slice(0, stop-start)})
    else:
        return data.sel(**{dim: dim_slice})

def get_cos_projection(vza, dx, x_center, _data, offset=0):
    """Apply cosine projection based on the viewing zenith angle (VZA).

    Args:
        vza (float): Viewing zenith angle in degrees.
        dx (float): Grid spacing in the x direction.
        x_center (xarray.DataArray): X-center of the cloud or data array.
        data (xarray.DataArray): Input dataset to project.

    Returns:
        xarray.DataArray: cos-projected dataset.
    """
    # NOTE: This could be reduced by upsampling by a smaller factor?
    upsampling_factor = 100. # upsampling factor to avoid interpolation artifacts
    x_up = np.linspace(_data.x.data[0], _data.x.data[-1], int(upsampling_factor) * len(_data.x)) # upsample x grid
    data = _data.interp(x=x_up) # interpolate data to upsampled x grid
    factor = np.cos(np.deg2rad(vza)) # cosine factor for projection = 0.5 for 60 deg
    dx_proj = dx/upsampling_factor * factor # in km, correct pixel spacing by cosine and upsampling factor
    xmax = data.x.data.max() # get max x value
    xmin = data.x.data.min() # get min x value
    x_proj = np.arange(xmin, xmax+dx_proj, dx_proj)
    nx = len(data.x) # original number of x points = 26500
    nx_proj = len(x_proj) # projected number of x points = 79089
    # pad xarray to match projected bin size
    pad = nx_proj - nx # amount of padding needed = 52589
    pad_before = pad//2 # pad before = 26294
    pad_after = pad - pad_before # pad after = 26295
    data_pad = data.pad({"x": (pad_before, pad_after)}, constant_values=0) # pad with zeros
    data_proj = data_pad.assign_coords({"x": x_proj}) # assign new x coordinates
    # interpolate to x grid from nadir view
    data_new = data_proj.interp(x=(x_center+offset)) # interpolate to original x grid center
    return data_new

def find_cosine_window(vza, x_range_an, dx, stretch=100):
    """Find the window for cosine projection based on viewing zenith angle (VZA).

    Args:
        vza (float): Viewing zenith angle in degrees.
        x_range_an (numpy.ndarray or list): Range of x values for analysis. This is typically the x-coordinates (in km) of the nadir view.
        dx (float): Grid spacing in the x direction.
        stretch (int, optional): Stretch factor to adjust the window size. Defaults to 100.

    Returns:
        slice: A slice object representing the cosine window.
    """
    factor = np.cos(np.deg2rad(vza)) # cosine factor for projection = 0.5 for 60 deg
    size = x_range_an.max() - x_range_an.min()
    radius = size/2. # radius of cloud region = 0.43999
    x_center = x_range_an.min() + radius
    #print(f"x_center of cosine window {x_center}")
    window = slice(x_center - radius/factor-(stretch*dx), x_center+radius/factor+(stretch*dx)) # in km
    return window


def get_projection_sortIndex(viewing_zenith_angle, delta_z, grid, grid_resolution):
    """ Get sort index to apply to original cloud image (registered to the ground)
    to obtain a projection to altitude delta_z [km] above ground.

    Args:
        viewing_zenith_angle: angle in degrees
        delta_z: altitude difference for projection, up is positive
        grid (list or np.nd-array): range of horizontal coordinates (1D) in direction of projection
        grid_resolution: resolution of horizontal grid
        verbose (bool): set to True to print debugging info

    Returns:
        sort_idx (int): index array to apply to data array to obtain projection
    """
    tan_factor = np.tan(np.deg2rad(-viewing_zenith_angle))
    shift = tan_factor * delta_z
    grid_new = grid + shift
    if shift < 0:
        mask = grid_new < grid.min()
        grid_new[mask] = ((grid.max() + grid_resolution + grid_new[mask]) % (grid.max() + grid_resolution))
        #print(viewing_zenith_angle, delta_z, shift, set((grid.max() + grid_resolution + grid_new[mask]) // (grid.max() + grid_resolution)))
    elif shift > 0:
        mask = grid_new > (grid.max() + grid_resolution)
        grid_new[mask] = grid_new[mask] % (grid.max() + grid_resolution)
        #print(viewing_zenith_angle, delta_z, shift, set((grid_new[mask]) // (grid.max())), set(np.diff(np.argsort(grid_new))))
        #print(((grid+shift)[mask] % (grid.max() + grid_resolution))[-1] - (grid+shift)[0])
    # sort index along shifted and wrapped grid axis
    sort_idx = np.argsort(grid_new)
    return sort_idx, grid_new

def generate_view_alternative(data, delta_z, slx, sly, offset=0, dx=0.04):
    x = data.x.data
    y = data.y.data
    vza = data.vza.data
    nx = len(x)
    # select cloud in nadir view
    nadir = data.sel(vza=0)[slx,sly]
    size = nadir.x.data.max() - nadir.x.data.min()
    radius = size/2.
    x_center = nadir.x.data.min() + radius
    cameras = []
    for _vza in vza:
        if _vza == 0:
            cameras.append(xr.DataArray(nadir.data, dims=["x", "y"], coords={"x": nadir.x.data, "y": nadir.y.data}))
        else:
            tan_factor = np.tan(np.deg2rad(_vza))
            shift = tan_factor * delta_z
            repeat = int(abs(shift)//x.max())
            n_repeat = int(3 + repeat)
            x_shift = x + shift
            print(_vza, shift, n_repeat)
            arrays = []
            for i in range(n_repeat):
                arrays.append(data.sel(vza=_vza).data)
            array_concat = np.concat(arrays, axis=0)
            #x_ext = np.arange(0, array_concat.shape[0]*dx, dx)
            # TODO: append negative, neutral, positive x
            x_ext = np.linspace(0, x.max()*n_repeat, array_concat.shape[0])
            extended = xr.DataArray(array_concat, dims=["x", "y"], coords={"x": x_ext, "y": y})
            od_proj = extended.interp(x=(repeat*x.max() + x_shift))
            od_proj = od_proj.assign_coords({"x": x})

            # cos projection
            factor = np.cos(np.deg2rad(_vza))
            dx_proj = dx * factor
            xmax = od_proj.x.data.max()
            xmin = od_proj.x.data.min()
            x_proj = np.arange(xmin, xmax+dx_proj, dx_proj)
            nx = len(od_proj.x)
            nx_proj = len(x_proj)
            # pad xarray to match projected bin size
            pad = nx_proj - nx
            pad_before = pad//2
            pad_after = pad - pad_before
            data_pad = od_proj.pad({"x": (pad_before, pad_after)}, constant_values=0)
            data_proj = data_pad.assign_coords({"x": x_proj})
            # interpolate to nadir window
            data_new = data_proj.interp(x=(x_center+offset))
            cameras.append(data_new)
    views = xr.concat(cameras, dim=xr.DataArray(vza, dims="vza", name="vza"))
    return views

def generate_views(_data, slx, sly, dx, cloud_COM=1.5, angles=[-60,0,60], offset=[]):
    data = project2com(_data, cloud_COM, dx) # project to cloud COM for parallex correction
    # select nadir view
    nadir = data.sel(vza=0)[slx, sly]
    # loop over off-nadir views and correct for periodic BCs
    cameras = []
    for ivza, vza in enumerate(angles):
        if vza == 0:
            cameras.append(nadir)
        else:
            # NOTE: this part crops the quivalent of the cloud region in the off-nadir views
            # same window for +/-vza since all 9 views are around cloud COM
            x_slice = find_cosine_window(abs(vza), nadir.x.data, dx) # calculates the window size needed to cover the cloud region based on cosine projection
            # wrap around through rolling in case the slice goes beyond domain boundaries
            # select the off-nadir view within the cosine window
            off_nadir = wrap_around(x_slice, data.sel(vza=vza), dx)[:,sly]
            off_nadir_proj = get_cos_projection(vza, dx, nadir.x, off_nadir, offset[ivza])
            cameras.append(off_nadir_proj.data)
            #cameras.append(off_nadir)
    #multi_angle_tile = cameras #np.stack(cameras, axis=-1)
    multi_angle_tile = np.stack(cameras, axis=-1)
    sinogram = xr.DataArray(dims=["x","y","angles"], data=multi_angle_tile, coords={"x": nadir.x, "y": nadir.y, "angles": angles})
    return sinogram


def plot_views(multi_angle_tile, angles, save_fname=None):
    fig, ax = plt.subplots(1, len(angles), figsize=(12,4), dpi=120)
    for i, vza in enumerate(angles):
        ax[i].set_aspect("equal")
        #ax[i].pcolormesh(multi_angle_tile[i].T)
        #multi_angle_tile[i].plot(x="x", add_colorbar=False)
        ax[i].pcolormesh(multi_angle_tile[...,i].T)
        ax[i].axis("off")
        ax[i].set_title(f"{vza:.0f} deg")
    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname)
    plt.show()

