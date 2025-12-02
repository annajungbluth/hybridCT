import numpy as np
from skimage.transform import radon

def get_diameter(data, pad=0):
    """
    Computes the diameter of the circumference of `data` in pixels.

    Args:
        data (numpy.ndarray): A 2D data array

    Returns:
        diameter (int)
    """
    nx, nz = data.shape
    max_dim = np.max([nx, nz])
    diameter = int(np.sqrt(2 * max_dim**2) + pad)
    return diameter


def pad_to_circle(image):
    diagonal = np.sqrt(2) * max(image.shape)
    pad = [int(np.ceil(diagonal - s)) for s in image.shape]
    new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
    old_center = [s // 2 for s in image.shape]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    return padded_image, diagonal


def _sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)

def crop_back(image, image_pad):
    width, height = image.shape
    excess_w = int(np.ceil((image_pad.shape[0] - width) / 2))
    excess_h = int(np.ceil((image_pad.shape[1] - height) / 2))
    s = np.s_[excess_w : width + excess_w, excess_h : height + excess_h]
    image_crop = image_pad[s]
    # Find the reconstruction circle, set reconstruction to zero outside
    #c0, c1 = np.ogrid[0:width, 0:width]
    #r = np.sqrt((c0 - width // 2) ** 2 + (c1 - width // 2) ** 2)
    #radius = min(image.shape) // 2
    #image_crop[r > radius] = 0.0
    return image_crop

def crop_back_sinogram(sino, sino_pad):
    width = sino.shape[0]
    excess_x = int(np.ceil((sino_pad.shape[0] - width) / 2))
    s = np.s_[excess_x : width + excess_x, :]
    sino_crop = sino_pad[s]
    return sino_crop

def pad_circle(data, pad=0):
    """
    Pads the input field with zeros to fit it into a square array
    which encloses the reconstruction circle of the ROI.

    Args:
        data (numpy.ndarray): A 2D array to be padded.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The padded 2D array.
            - int: The diameter of the reconstruction circle.

    Notes:
        Assumes center of reconstruction circle is at the center
        of the input `data` field.
    """
    diameter = get_diameter(data, pad)
    data_pad = np.zeros((diameter,diameter))
    nz, nx = data.shape
    nz_pad, nx_pad = data_pad.shape
    x_start = nx_pad//2-nx//2
    x_window = slice(x_start, x_start+nx)
    z_start = nz_pad//2-nz//2
    z_window = slice(z_start, z_start+nz)
    data_pad[z_window, x_window] = data
    return data_pad, diameter


def apply_open_boundary_conditions(thetas, data, window_xz):
    """
    Expand `data` field assuming open boundary conditions based on the
    maximum absolute angle (`thetas`) to ensure the entire view is covered.
    It expands the data array by padding the field with zeros and adjusts
    the slice object `slx` accordingly.

    Args:
        thetas (array-like): Array of angles in degrees.
        data (ndarray): 2D array of data to which boundary conditions are applied.
        window_xz (tuple): Tuple of slice objects (`slx`, `slz`) defining the ROI, defined
                            as `slx = slice(x_start, x_stop)`, `slz = slice(z_base, z_top)`.

    Returns:
        tuple: A tuple containing two elements:
            - ndarray: Data array padded with zeros along the x-axis.
            - slice: `slx` original ROI adjusted for offset.
    """
    # find most oblique view
    view_max = np.max(abs(thetas))
    slx, slz = window_xz
    data_window = data[slz, slx]
    nz, nx = data_window.shape
    delta_z = slz.stop - slz.start
    n_pad = round(np.tan(np.deg2rad(view_max)) * delta_z)
    nx_pad = nx + 2 * n_pad
    data_pad = np.zeros((nz, nx_pad))
    cx = nx // 2
    cx_pad = nx_pad // 2
    slx_shifted = slice(cx_pad - cx, cx_pad - cx + nx)
    data_pad[:, slx_shifted] = data_window
    return data_pad, slx_shifted


def apply_periodic_boundary_conditions(thetas, data, window_xz, verbose=False):
    """
    Expand `data` field based on the maximum absolute angle (`thetas`) to
    ensure the entire view is covered. It expands the data array by
    replicating the field using periodic boundary conditions and adjusts
    the slice object `slx` accordingly.

    Args:
        thetas (array-like): Array of angles in degrees.
        data (ndarray): 2D array of data to which boundary conditions are applied.
        window_xz (tuple): Tuple of slice objects (`slx`, `slz`) defining the ROI, defined
                            as `slx = slice(x_start, x_stop)`, `slz = slice(z_base, z_top)`.
        verbose (bool): flag to enable/disable comments (default: False).

    Returns:
        tuple: A tuple containing two elements:
            - ndarray: Padded data array with periodic boundary conditions applied along the x-axis.
            - slice: `slx` original ROI adjusted for offset.
    """
    # find most oblique view
    view_max = np.max(abs(thetas))
    slx, slz = window_xz
    delta_z = slz.stop - slz.start
    n_pad = round(np.tan(np.deg2rad(view_max)) * delta_z)
    # full dimensions of input data
    nz, nx = data.shape
    # canvas size to fit in most oblique view
    n_wrap_right = slx.stop + n_pad
    n_wrap_left = slx.start - n_pad
    data_pad = data.copy()
    offset = 0
    if n_wrap_left < 0:
        # how many pixels are we overshooting to the left?
        overshoot_left = abs(n_wrap_left)
        if verbose:
            print(f"overshooting left by {overshoot_left} pixels")
        # how many times do we have to replicate the scene?
        n_periods = int(np.ceil(overshoot_left / nx))
        for i in range(n_periods):
            data_pad = np.hstack((data_pad, data))
        offset = (i+1) * nx
    slx_shifted = slice(slx.start - n_wrap_left, slx.stop - n_wrap_left)
    slx_pad = slice(n_wrap_left + offset, n_wrap_right + offset)

    nz_pad, nx_pad = data_pad.shape
    if slx_pad.stop > nx_pad:
        # how many pixels are we overshooting to the right?
        overshoot_right = abs(nx_pad - slx_pad.stop)
        if verbose:
            print(f"overshooting right by {overshoot_right} pixels")
        # how many times do we have to replicate the scene?
        n_periods = int(np.ceil(overshoot_right / nx))
        for i in range(n_periods):
            data_pad = np.hstack((data_pad, data))
    if verbose:
        print(f"test: needed padding {n_pad}, computed left {slx_shifted.start}, right {data_pad[:,slx_pad].shape[1] - slx_shifted.stop}")
    return data_pad[:,slx_pad], slx_shifted

def compute_sinogram(data, thetas, config):
    # wv with pBC
    wv_data_pad_circ, diameter_pad = pad_circle(wv_data_pad)
    # wv w/o pBC
    wv_data_circ, diameter = pad_circle(wv_data[slz,slx])
    # find rotation center for cropping sinogram back to window w/o pBC
    radius_offset = int((diameter_pad - diameter)/2)
    crop = slice(radius_offset, diameter_pad - radius_offset - 1)
    sino_bc = radon(wv_data_pad_circ, theta=theta)
    return sino_bc[crop]
