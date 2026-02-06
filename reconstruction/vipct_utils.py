import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

def generate_cloud_file_dataset_vipct(data_dict, total_pad=0, flip_x=False, flip_y=False, flip_xy=False):
    """
    Function to generate cloud file dataset from VIPCT data.
    Parameters:
    data_dict: dict
        Dictionary containing 'ext' and 'grid' keys.
    dx: float
        Grid spacing in km.
    total_pad: int
        Number of pixels to add around the data.
    flip_x: bool
        Whether to flip the x-axis of the extinction data.
    flip_y: bool
        Whether to flip the y-axis of the extinction data.
    flip_xy: bool
        Whether to swap the x and y axes of the extinction data.
    Returns:
    xarray.Dataset
        Dataset containing cloud extinction and grid information.
    """
    ext_ = data_dict['ext']

    if flip_x:
        ext_ = ext_[::-1, :, :]
    if flip_y:
        ext_ = ext_[:, ::-1, :]
    if flip_xy:
        ext_ = ext_.transpose(1, 0, 2) # swap x and y axes

    nx, ny, nz = ext_.shape

    dx = data_dict['grid'][0][1] - data_dict['grid'][0][0]  # assuming uniform spacing
    dy = data_dict['grid'][1][1] - data_dict['grid'][1][0]  # assuming uniform spacing
    dz = data_dict['grid'][2][1] - data_dict['grid'][2][0]  # assuming uniform spacing

    # add padding
    ext = np.zeros((nx+total_pad, ny+total_pad, nz)) # padded LWC array

    pad_l = np.floor(total_pad/2).astype(int)
    pad_r = np.ceil(total_pad/2).astype(int)

    slx = slice(pad_l, pad_l+nx) # slice for x dimension with padding
    sly = slice(pad_l, pad_l+ny) # slice for y dimension with padding

    ext[slx,sly,:] = ext_.data # insert original extinction data

    x = np.concatenate((np.arange(-pad_l*dx, 0, dx), np.arange(0, nx*dx, dx), np.arange(nx*dx, (nx+pad_r)*dx, dx))) # define x-coordinates with padding
    y = np.concatenate((np.arange(-pad_l*dy, 0, dy), np.arange(0, ny*dy, dy), np.arange(ny*dy, (ny+pad_r)*dy, dy))) # define y-coordinates with padding
    z = data_dict['grid'][2][:nz] # define z-coordinates

    ds = xr.Dataset({
        "ext": xr.DataArray(ext, dims=["x", "y", "z"], coords=[x, y, z]),
        "delx": dx,
        "dely": dy,
        "delz": dz,
        })
    
    return ds # returns padded dataset

def extract_viewing_angles(data, idx):
    """Extract VZA and azimuth from camera rotation matrix.
    
    Args:
        data: Dict with 'rotation' matrix
        idx: Index of the camera to extract angles for
    
    Returns:
        vza: Viewing zenith angle in degrees
        azimuth: Azimuth angle in degrees
    """
    R = data['cameras_R'][idx]
    
    # Camera optical axis is third row of R (or third column of R^T)
    # This points in the viewing direction
    viewing_direction = R[2, :]  # [dx, dy, dz]
    
    # VZA: angle from nadir (vertical down = [0, 0, -1])
    # cos(vza) = dot(viewing_dir, [0,0,-1]) = -viewing_dir[2]
    cos_vza = -viewing_direction[2] # negative because viewing down
    vza = np.rad2deg(np.arccos(np.clip(cos_vza, -1, 1)))
    
    # Azimuth: rotation around vertical axis
    # Projected viewing direction in horizontal plane
    azimuth = np.rad2deg(np.arctan2(viewing_direction[1], viewing_direction[0]))

    # Convert to signed angle based on azimuth direction
    reference_azimuth = 135.00 # the azimuth of the nadir view
    azimuth_diff = (azimuth - reference_azimuth + 180) % 360 - 180
    vza_signed = vza if abs(azimuth_diff) < 90 else -vza
    
    return vza, azimuth, vza_signed


def rectify_with_projection_matrix(image, P, cloud_height, output_shape, 
                                    x_range, y_range):
    """
    Rectify/parallax-correct an image using the projection matrix.
    
    Args:
        image: Original camera image [H, W]
        P: Projection matrix [3, 4] for this camera
        cloud_height: Height (km) at which to align features
        output_shape: (H_out, W_out) for rectified image
        x_range: (x_min, x_max) world coordinates in km
        y_range: (y_min, y_max) world coordinates in km
    
    Returns:
        Rectified image
    """
    H_out, W_out = output_shape
    H_in, W_in = image.shape
    
    # Create output world coordinate grid at cloud_height
    x_world = np.linspace(x_range[0], x_range[1], W_out)
    y_world = np.linspace(y_range[0], y_range[1], H_out)
    X, Y = np.meshgrid(x_world, y_world)
    
    # Create homogeneous world coordinates at cloud height
    # Shape: (H_out * W_out, 4)
    ones = np.ones_like(X)
    Z = ones * cloud_height
    world_points = np.stack([X, Y, Z, ones], axis=-1).reshape(-1, 4)
    
    # Project all world points to camera image coordinates
    # proj = P @ [x, y, z, 1]^T → [u, v, w]
    proj = (P @ world_points.T).T  # (N, 3)
    
    # Normalize and swap (to match VIPCT convention)
    # Converts from homogeneous ['u', 'v', 'w'] to normalized coords ['u'/'w', 'v'/'w'] in range [-1, 1]
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    
    # Convert normalized [-1, 1] to pixels, with axis swap
    px = (v + 1) * W_in / 2  # v → x (horizontal), remaps -1 -> 0, 0 -> 58, +1 -> 116
    py = (u + 1) * H_in / 2  # u → y (vertical)
    
    # Reshape to output grid
    # this tells us what pixels (i,j) to sample
    px = px.reshape(H_out, W_out)
    py = py.reshape(H_out, W_out)
    
    # Sample original image using these coordinates using bilinear interpolation
    from scipy.ndimage import map_coordinates
    rectified = map_coordinates(image, [py, px], order=1, mode='nearest', cval=0)
    
    return rectified

def create_aligned_views(images, data_dict, cloud_height, output_shape=None, padding_factor=1, resolution=0.02):
    """Create parallax-corrected views using projection matrices."""
    
    n_cams = images.shape[0]
    
    if output_shape is None:
        output_shape = images[0].shape

    # Expand world coordinate range to capture parallax-shifted content
    x_size = 1.55 # output_shape[1] * resolution
    y_size = 1.55 # output_shape[0] * resolution
    
    # Add padding to account for parallax shift
    x_pad = x_size * (padding_factor - 1) / 2
    y_pad = y_size * (padding_factor - 1) / 2
    
    x_range = (-x_pad, x_size + x_pad)
    y_range = (-y_pad, y_size + y_pad)
    
    aligned_views = []
    
    for i in range(n_cams):
        P = data_dict['cameras_P'][i]
        img = images[i]
        
        rectified = rectify_with_projection_matrix(
            image=img,
            P=P,
            cloud_height=cloud_height,
            output_shape=output_shape,
            x_range=x_range,
            y_range=y_range
        )
        aligned_views.append(rectified)

    # Stack aligned views into a multi-angle tile
    multi_angle_tile = np.stack(aligned_views, axis=-1) # shape (H, W, n_cams)

    sinogram = xr.DataArray(dims=["x","y","angles"], data=multi_angle_tile, coords={"x": images.x.values, "y": images.y.values, "angles": images.vza.values})

    return sinogram

def coarsen_image_resolution(dataset, res, coarse_res):
    """Return coarsened dataset based on original resolution `res`
    to coarse resolution `coarse_res`.
    """
    factor = coarse_res / res
    print(f"Coarsening dataset using factor {factor}")
    
    # Calculate new coordinates
    new_x = np.arange(dataset.x[0], dataset.x[-1]+res, coarse_res)
    new_y = np.arange(dataset.y[0], dataset.y[-1]+res, coarse_res)
    
    # Interpolate to new grid
    coarse = dataset.interp(x=new_x, y=new_y, method='cubic')
    return coarse

def upsample_volume(volume, new_res):
    """
    Upsample volume using chunked interpolation to reduce memory usage.
    
    Args:
        volume: xarray Dataset with 'ext' variable
        new_res: target resolution in km
        chunk_size: number of slices to process at once (reduce if still OOM)
    """
    x_grid = volume.x.data
    y_grid = volume.y.data
    z_grid = volume.z.data

    original_coords = (x_grid, y_grid, z_grid)
    
    # Create new coordinate arrays
    x_grid_new = np.arange(x_grid[0], x_grid[-1], new_res)
    y_grid_new = np.arange(y_grid[0], y_grid[-1], new_res)
    
    nx_new = len(x_grid_new)
    ny_new = len(y_grid_new)
    nz = len(z_grid)
    
    # Pre-allocate output array
    upsampled_volume = np.zeros((nx_new, ny_new, nz))
    
    # Create interpolator once
    interp_func = RegularGridInterpolator(original_coords, volume.ext.data, 
                                          method='linear')
    
    # Process in z-slices or xy-chunks
    for z_idx in range(nz):
        # Create 2D meshgrid for this z-slice only
        X_slice, Y_slice = np.meshgrid(x_grid_new, y_grid_new, indexing='ij')
        Z_slice = np.full_like(X_slice, z_grid[z_idx])
        
        # Stack coordinates for this slice
        coords_slice = np.stack([X_slice.ravel(), Y_slice.ravel(), Z_slice.ravel()], axis=-1)
        
        # Interpolate this slice
        upsampled_volume[:, :, z_idx] = interp_func(coords_slice).reshape(nx_new, ny_new)

    scaling_factor = volume.ext.data.sum() / upsampled_volume.sum()
    upsampled_volume *= scaling_factor
    
    # Create output dataset
    upsampled_ds = xr.Dataset({
        "ext": xr.DataArray(upsampled_volume, dims=["x", "y", "z"], 
                           coords=[x_grid_new, y_grid_new, z_grid]),
        "delx": new_res,
        "dely": new_res,
        "delz": volume.delz,
    })
    
    return upsampled_ds