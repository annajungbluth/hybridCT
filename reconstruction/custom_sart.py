from skimage.transform import order_angles_golden_ratio, warp
import numpy as np
from math import cos, sin, floor, ceil, sqrt, pi
from numba import jit

def custom_radon(image, theta, resolution, rotation_center=None):

    # shape of the input image is (66, 66), which is (nz, nx)
    # theta = [-70.5, -60. , -45.6, -26.1,   0. ,  26.1,  45.6,  60. ,  70.5]
    # resolution = 0.04
    # rotation_center = None
    radon_image = np.zeros((image.shape[0], len(theta)), dtype=image.dtype)
    if rotation_center is None:
        center = image.shape[0] // 2 # center = 33
    else: 
        center = rotation_center

    for i, angle in enumerate(np.deg2rad(theta)): # loop over 9 angles
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array(
            [
                [cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                [0, 0, 1],
            ]
        )
        rotated = warp(image, R, clip=False) * resolution
        radon_image[:, i] = rotated.sum(0) # sum over all height (nz)
    return radon_image # shape (66, 9), equivalent to (n_detectors (x-pixels), n_angles)


def custom_radon_ray_sum(image, theta, resolution, rotation_center=None):
    n_theta = len(theta)
    sinogram = np.zeros((image.shape[0], n_theta))
    mask = np.ones(image.shape, dtype="bool")
    for itheta, _theta in enumerate(theta):
        for ray_pos in range(image.shape[0]):
            ray_sum, weight_norm = bilinear_ray_sum(image, mask, _theta, ray_pos, resolution=resolution, rotation_center=rotation_center)
            sinogram[ray_pos, itheta] = ray_sum
    return sinogram


def iradon_sart_custom(
    radon_image,
    theta=None,
    image=None,
    mask=None,
    clip=None,
    rotation_center=None,
    relaxation=0.15,
    resolution=1.,
    apply_hamming_window=False
):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using a single iteration of
    the Simultaneous Algebraic Reconstruction Technique (SART) algorithm.

    Parameters
    ----------
    radon_image : ndarray, shape (M, N)
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array, shape (N,), optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    image : ndarray, shape (M, M), optional
        Image containing an initial reconstruction estimate. Default is an array of zeros.
    projection_shifts : array, shape (N,), optional
        Shift the projections contained in ``radon_image`` (the sinogram) by
        this many pixels before reconstructing the image. The i'th value
        defines the shift of the i'th column of ``radon_image``.
    clip : length-2 sequence of floats, optional
        Force all values in the reconstructed tomogram to lie in the range
        ``[clip[0], clip[1]]``
    relaxation : float, optional
        Relaxation parameter for the update step. A higher value can
        improve the convergence rate, but one runs the risk of instabilities.
        Values close to or higher than 1 are not recommended.
    dtype : dtype, optional
        Output data type, must be floating point. By default, if input
        data type is not float, input is cast to double, otherwise
        dtype is set to input data type.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    Algebraic Reconstruction Techniques are based on formulating the tomography
    reconstruction problem as a set of linear equations. Along each ray,
    the projected value is the sum of all the values of the cross section along
    the ray. A typical feature of SART (and a few other variants of algebraic
    techniques) is that it samples the cross section at equidistant points
    along the ray, using linear interpolation between the pixel values of the
    cross section. The resulting set of linear equations are then solved using
    a slightly modified Kaczmarz method.

    When using SART, a single iteration is usually sufficient to obtain a good
    reconstruction. Further iterations will tend to enhance high-frequency
    information, but will also often increase the noise.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] AH Andersen, AC Kak, "Simultaneous algebraic reconstruction
           technique (SART): a superior implementation of the ART algorithm",
           Ultrasonic Imaging 6 pp 81--94 (1984)
    .. [3] S Kaczmarz, "Angenäherte auflösung von systemen linearer
           gleichungen", Bulletin International de l’Academie Polonaise des
           Sciences et des Lettres 35 pp 355--357 (1937)
    .. [4] Kohler, T. "A projection access scheme for iterative
           reconstruction based on the golden section." Nuclear Science
           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.
    .. [5] Kaczmarz' method, Wikipedia,
           https://en.wikipedia.org/wiki/Kaczmarz_method

    """
    # the image is the reconstruction estimate (or prior), with shape (nz, nx), i.e. (66, 66)
    if image is None:
        image = np.zeros((radon_image.shape[0],radon_image.shape[0])) # if no prior, start from zeros
    if mask is None:
        mask = np.ones((radon_image.shape[0],radon_image.shape[0]), dtype="bool") # if no mask, all True with shape (nz, nx), i.e. (66, 66)
    dtype = np.float32
    #if projection_shifts is None:
    projection_shifts = np.zeros((radon_image.shape[1]), np.float32) # shape (n_angles,), all zeros

    reconstructed_shape = (radon_image.shape[0], radon_image.shape[0]) # shape (nz, nx), i.e. (66, 66)

    for angle_index in order_angles_golden_ratio(theta): # organizes angles in golden ratio order to support faster SART convergence
        image_update = sart_projection_update( # processes in order [0, ]
            image, # shape (nz, nx), i.e. (66, 66). This is the current reconstruction estimate
            mask, # shape (nz, nx), i.e. (66, 66)
            theta[angle_index],
            radon_image[:, angle_index], # shape (n_detectors,), i.e. (66,)
            rotation_center, # usually None
            projection_shifts[angle_index], # usually 0.
            resolution, # resolution of 0.04
            apply_hamming_window, # usually False
        ) # returns the correction to the reconstruction estimate from this projection angle, shape (nz, nx), i.e. (66, 66)
        image += relaxation * image_update # the relaxation prevents overshooting and oscillations. This makes converge slower but more stable
        if clip is not None:
            image = np.clip(image, clip[0], clip[1])
    return image

@jit
def bilinear_ray_sum(image, mask, _theta, ray_position, resolution, rotation_center=None):
    """
    Compute the projection of an image along a ray.

    Parameters
    ----------
    image : ndarray of float, shape (M, N)
        Image to project.
    theta : float
        Angle of the projection.
    ray_position : float
        Position of the ray within the projection.

    Returns
    -------
    projected_value : float
        Ray sum along the projection.
    norm_of_weights :
        A measure of how long the ray's path through the reconstruction circle was.
    """
    # this function is the forward projection operation using bilinear interpolation
    theta = _theta / 180. * pi # convert to radians
    radius = image.shape[0] // 2 - 1 # image.shape[0] = 66, radius = 32, reconstruction circle radius
    projection_center = image.shape[0] // 2 # projection center = 33
    if rotation_center is None:
        rotation_center = image.shape[0] // 2 # rotation center = 33
    t = ray_position - projection_center # offset from center, ray_position goes from 0 to 65
    s0 = sqrt(radius * radius - t * t) if radius * radius >= t * t else 0. # half-length of ray within circle
    Ns = 2 * ceil(2 * s0)  # number of steps along the ray
    ray_sum = 0. 
    weight_norm = 0.

    if Ns > 0:
        ds = 2 * s0 / Ns # step size along the ray
        dx = -ds * cos(theta) # step size in x
        dy = -ds * sin(theta) # step size in y
        x0 = s0 * cos(theta) - t * sin(theta) # x-coordinate of where the ray enters the circle
        y0 = s0 * sin(theta) + t * cos(theta) # y-coordinate of where the ray enters the circle
        for k in range(int(Ns) + 1): # loop through each step along the ray
            """
            Simple pseudocode for bilinear interpolation ray sum:

            ray_sum = 0
            for each sample point along the ray:
                value_at_this_point = bilinear_interpolate(image, position)
                ray_sum += value_at_this_point
            return ray_sum

            This is calculating the (discreet) line integral along the ray through the image.
            Mathematically: ray_sum ≈ ∫_ray β(s) ds
            Why bilinear instead of just nearest pixel?
                - Smooth, sub-pixel accuracy
                - Reduces aliasing artifacts
                - More accurate line integrals
            """
            x = x0 + k * dx # get the x-coordinate at step k
            y = y0 + k * dy # get the y-coordinate at step k
            index_i = x + rotation_center # convert to image index
            index_j = y + rotation_center # convert to image index
            i = int(floor(index_i)) # integer part (whole pixel row)
            j = int(floor(index_j)) # integer part (whole pixel column)
            di = index_i - floor(index_i) # decimal part (how far into the pixel row)
            dj = index_j - floor(index_j) # decimal part (how far into the pixel column)
            if mask[i,j] == True: # only accumulate if within mask
                # needs to calculate weighted sum, since we might be between 4 neighboring pixels
                if i > 0 and j > 0: 
                    # mutliply by ds to account for step size along ray, and resolution to convert to physical units
                    weight = (1. - di) * (1. - dj) * ds * resolution # weight for top-left pixel
                    ray_sum += weight * image[i, j]
                    weight_norm += weight * weight
                if i > 0 and j < image.shape[1] - 1:  # weight for top-right pixel
                    weight = (1. - di) * dj * ds * resolution
                    ray_sum += weight * image[i, j+1]
                    weight_norm += weight * weight
                if i < image.shape[0] - 1 and j > 0: # weight for bottom-left pixel
                    weight = di * (1 - dj) * ds * resolution
                    ray_sum += weight * image[i+1, j]
                    weight_norm += weight * weight 
                if i < image.shape[0] - 1 and j < image.shape[1] - 1: # weight for bottom-right pixel
                    weight = di * dj * ds * resolution
                    ray_sum += weight * image[i+1, j+1]
                    weight_norm += weight * weight

    return ray_sum, weight_norm # the weight_norm is used to normalize rays, central rays are longer than edge rays

@jit
def bilinear_ray_update(image, _image_update, theta, ray_position, deviation, resolution, apply_hamming_window, rotation_center=None):
    """Compute the update along a ray using bilinear interpolation.

    Parameters
    ----------
    image : ndarray of float, shape (M, N)
        Current reconstruction estimate.
    image_update : ndarray of float, shape (M, N)
        Array of same shape as ``image``. Updates will be added to this array.
    theta : float
        Angle of the projection.
    ray_position : float
        Position of the ray within the projection.

    Returns
    -------
    deviation :
        Deviation before updating the image.



    Addiitional Notes:
    -----------
    Backprojection (bilinear_ray_update):
        - Walk along the same ray as in bilinear_ray_sum
        - Write corrections to pixels
        - Distribute deviation (error) back to all pixels the ray passed through
        - Positive deviation -> increase pixel values
        - Negative deviation -> decrease pixel values
        - Not equal distribution:
            - Within each sample point: split unequally among 4 neighbors (bilinear)
            - Along the ray: usually equal, but reduced at endpoints if Hamming is enabled
        - But this ignores the mask, and distributes to all pixels along the ray
        - The mask is applied later in sart_projection_update to zero out updates outside the mask
        - Result: pixels at the center of the ray path get more correction than those at edges, and within each sample location, closer pixels get more weight than farther ones.
    """

    theta = theta / 180. * pi # convert to radians
    radius = image.shape[0] // 2 - 1 # reconstruction circle radius
    projection_center = image.shape[0] // 2 # projection center
    if rotation_center is None:
        rotation_center = image.shape[0] // 2 # rotation center
    # ray position relative to center
    t = ray_position - projection_center
    # 2*s0 = distance to raytrace within circle of radius
    s0 = sqrt(radius * radius - t * t) if radius * radius >= t * t else 0.
    # total distance to raytrace in pixels
    Ns = 2 * ceil(2 * s0) 
    hamming_beta = 0.46164 # standard value for Hamming window
    hamming_window = 1. # default value if not applying Hamming window

    if Ns > 0: # walk along the ray
        # step size, slant path
        ds = 2 * s0 / Ns
        # step size in x and y
        dx = -ds * cos(theta)
        dy = -ds * sin(theta)
        # starting point in x and y pixel coordinates
        x0 = s0 * cos(theta) - t * sin(theta)
        y0 = s0 * sin(theta) + t * cos(theta)
        for k in range(int(Ns) + 1):
            x = x0 + k * dx
            y = y0 + k * dy
            index_i = x + rotation_center
            index_j = y + rotation_center
            i = int(floor(index_i))
            j = int(floor(index_j))
            di = index_i - floor(index_i)
            dj = index_j - floor(index_j)
            if apply_hamming_window: # usually False
                hamming_window = ((1 - hamming_beta) - hamming_beta * cos(2 * pi * k / (Ns - 1)))
            # update via bilinear interpolation
            if i > 0 and j > 0:
                _image_update[i, j] += (deviation * (1. - di) * (1. - dj)
                                       * ds * hamming_window) * resolution
            if i > 0 and j < image.shape[1] - 1:
                _image_update[i, j+1] += (deviation * (1. - di) * dj
                                         * ds * hamming_window) * resolution
            if i < image.shape[0] - 1 and j > 0:
                _image_update[i+1, j] += (deviation * di * (1 - dj)
                                         * ds * hamming_window) * resolution
            if i < image.shape[0] - 1 and j < image.shape[1] - 1:
                _image_update[i+1, j+1] += (deviation * di * dj
                                           * ds * hamming_window) * resolution
    return _image_update

@jit # compiles with numba in C-speed
def sart_projection_update(image, mask, theta, projection, rotation_center=None, projection_shift=0., resolution=1., apply_hamming_window=False):
    """
    Compute update to a reconstruction estimate from a single projection
    using bilinear interpolation.

    Parameters
    ----------
    image : ndarray of float, shape (M, N)
        Current reconstruction estimate
    theta : float
        Angle of the projection
    projection : ndarray of float, shape (P,)
        Projected values, taken from the sinogram
    projection_shift : float
        Shift the position of the projection by this many pixels before
        using it to compute an update to the reconstruction estimate

    Returns
    -------
    image_update : ndarray of float, shape (M, N)
        Array of same shape as ``image`` containing updates that should be
        added to ``image`` to improve the reconstruction estimate
    """
    dtype = np.float32 #if image.dtype == np.float32 else np.float64
    image_update = np.zeros_like(image, dtype=dtype) # image shape (nz, nx), i.e. (66, 66)
    # projection is the sinogram column for this angle, shape (n_detectors,), i.e. (66,)
    for ray_position in range(projection.shape[0]): # loop over each ray position, i.e. 66 positions. This assumes rays come straight through each pixel column
        # raytracing of prior guess
        # ray here just means "line integral path", not actual photon rays
        ray_sum, weight_norm = bilinear_ray_sum(image, mask, theta, ray_position, resolution, rotation_center)
        # compute deviation between sinogram of prior and measurement
        if weight_norm > 0.:
            deviation = -(ray_sum - projection[ray_position]) / weight_norm
        else:
            deviation = 0.
        # update
        image_update = bilinear_ray_update(image, image_update, theta, ray_position, deviation, resolution, apply_hamming_window, rotation_center)

        # apply mask to image update, and zero out updates outside the mask
        # NOTE: could the loop be replaced with image_update *= mask?
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] == False:
                    image_update[i,j] = 0
    return np.asarray(image_update)


