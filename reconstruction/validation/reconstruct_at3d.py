import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from collections import OrderedDict
import pylab as py
from scipy import stats
import warnings
np.random.seed(1)

def crop(cloud):
    left = np.where(cloud.sum(axis=0)>0)[0][0]
    right = np.where(cloud.sum(axis=0)>0)[0][-1]
    base = np.where(cloud.sum(axis=1)>0)[0][0]
    top = np.where(cloud.sum(axis=1)>0)[0][-1]
    slz = slice(base, top)
    slx = slice(left, right)
    return cloud[slz, slx], slz, slx

def fit2canvas(truth, rec, error, cloud_id):
    # load template
    cloud = np.load(f"data/true_slice_id{cloud_id}.npz")["truth"]
    cloud_crop, slz, slx = crop(cloud)
    truth_crop, slz_crop, slx_crop = crop(truth)
    # create new array
    truth_new = np.zeros_like(cloud)
    truth_new[slz, slx] = truth_crop
    rec_new = np.zeros_like(cloud)
    rec_new[slz, slx] = rec[slz_crop, slx_crop]
    err_new = np.zeros_like(cloud)
    err_new[slz, slx] = error[slz_crop, slx_crop]
    return truth_new, rec_new, err_new

def fit2canvas_profile(truth, rec, cloud_id):
    base_truth = np.where(truth>0)[0][0]
    top_truth = np.where(truth>0)[0][-1]
    delta_truth = top_truth - base_truth
    slz_crop = slice(base_truth, top_truth)
    cloud = np.load(f"data/true_slice_id{cloud_id}.npz")["truth"].sum(axis=1)
    base = np.where(cloud>0)[0][0]
    top = np.where(cloud>0)[0][-1]
    slz = slice(base, base+delta_truth)
    truth_new = np.zeros_like(cloud)
    truth_new[slz] = truth[slz_crop]
    rec_new = np.zeros_like(cloud)
    rec_new[slz] = rec[slz_crop]
    return truth_new, rec_new

def get_sample_indices(data,num_samples=5000, num_bins=100):
    bins = np.linspace(data.min(), data.max(), num_bins + 1)
    sampled_indices = []
    for i in range(num_bins):
        # Get indices of values in the current bin
        bin_mask = (data >= bins[i]) & (data < bins[i+1])
        bin_indices = np.where(bin_mask)[0]
        # How many to sample from this bin
        n = num_samples // num_bins
        # Avoid trying to sample more than exists in a bin
        if len(bin_indices) < n:
            n = len(bin_indices)
        if n > 0:
            sampled_indices.extend(np.random.choice(bin_indices, n, replace=False))
    return sampled_indices

def scatter_density(ax, _x, _y, x_name, y_name, save_dir=None, n_sample=10000, stratified=False):
    '''
    makes a scatter plot and color codes where most of the data is
    :param x: x-value
    :param y: y-value
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param save_dir: save location
    :return: -
    '''
    # print('making scatter density plot ... ')
    # need to reduce number of samples to keep processing time reasonable.
    # Reduce if processing time too long or run out of RAM
    # linear fit
    slope, offset = np.polyfit(_x,_y, 1)
    y_fit = [np.min(_x)*slope+offset, np.max(_x)*slope+offset]
    #max_n = 50000
    if n_sample > len(_x):
        n_sample = len(_x)
    #subsample = int(len(x) / max_n)
    #rand_idx = np.random.randint(0, len(x)-1, n_sample)
    if stratified:
        rand_idx = get_sample_indices(_x)
    else:
        rand_idx = np.random.choice(len(_x), n_sample, replace=False) #random.sample(x, n_sample)#x[::subsample]
    x = _x[rand_idx]
    y = _y[rand_idx]
    try:
        r, _ = stats.pearsonr(_x, _y)  # get R
    except:
        print('could not calculate r, set to nan')
        r = np.nan
    xy = np.vstack([x, y])
    if np.mean(x) == np.mean(y):
        z = np.arange(len(x))
    else:
        z = stats.gaussian_kde(xy)(xy)  # calculate density
    # sort points by density
    idx = z.argsort()
    d_feature = x[idx]
    d_target = y[idx]
    z = z[idx]
    # plot everything
    ax.grid(ls=":", zorder=-100)
    ax.scatter(_x, _y, c="0.9", s=2)
    ax.scatter(d_feature, d_target, c=z, s=2, cmap="Blues") #label=r'R$^2$ = ' + str(np.round(r, 2)), zorder=100)
    vmin = np.min([d_feature])#, d_target]) #np.percentile(d_feature,1), np.percentile(d_target,1)])
    vmax = np.max([d_feature])#, d_target]) #np.percentile(d_feature,99), np.percentile(d_target,99)])
    ax.set_xlim(vmin, vmax)
    return ax


if __name__ == "__main__":
    run_retrieval = False

    cloud_id = 55
    tag = "cloud_id055"
    time = 30.

    if run_retrieval:
        import at3d
        sensors, solvers, rte_grid = at3d.util.load_forward_model("results/radiances_phi0_wc_les_RICO_40m_80kmx80km_T_qc_{time:.1f}h_id{cloud_id:03d}_pad.nc")
        wvl = 0.67
        maxiter = 50
        cloud_mask = solvers[wvl].medium['cloud'].extinction.data > 0

        # Perform some cloud masking using a single fixed threshold based on the observation that
        # everywhere else will be very dark.
        sensor_list = []
        for sensor in sensors['MISR']['sensor_list']:
            copied = sensor.copy(deep=True)
            weights = np.zeros(sensor.sizes['nrays'])
            ray_mask =np.zeros(sensor.sizes['nrays'], dtype=int)
            ray_mask_pixel = np.zeros(sensor.npixels.size, dtype=int)
            ray_mask_pixel[np.where(sensor.I.data > 2e-3)] = 1
            copied['weights'] = ('nrays',sensor.I.data)
            copied['cloud_mask'] = ('nrays', ray_mask_pixel[sensor.pixel_index.data])
            sensor_list.append(copied)

        space_carver = at3d.space_carve.SpaceCarver(rte_grid, bcflag=3)
        carved_volume = space_carver.carve(sensor_list, agreement=(0.0, 1.0), linear_mode=False)
        # remove cloud mask values at outer boundaries to prevent interaction with open boundary conditions.
        carved_volume.mask[0] = carved_volume.mask[-1] =carved_volume.mask[:,0] =carved_volume.mask[:,-1] = 0.0
        # make forward_sensors which will hold synthetic measurements from the evaluation of the forward model.
        forward_sensors = sensors.make_forward_sensors()
        # add an uncertainty model to the observations.
        uncertainty = at3d.uncertainties.RadiometricNoiseUncertainty(1e-5, 1e-3)
        sensors.add_uncertainty_model('MISR', uncertainty)
        # prepare all of the static inputs to the solver just copy pasted from forward model
        surfaces = OrderedDict()
        numerical_parameters = OrderedDict()
        sources = OrderedDict()
        num_stokes = OrderedDict()
        background_optical_scatterers = OrderedDict()
        for key in forward_sensors.get_unique_solvers():
            surfaces[key] = solvers[key].surface
            numerical_params = solvers[key].numerical_params
            #numerical_params['num_mu_bins'] = 2
            #numerical_params['num_phi_bins'] = 4
            numerical_parameters[key] = numerical_params
            sources[key] = solvers[key].source
            num_stokes[key] = solvers[key]._nstokes
            background_optical_scatterers[key] = {'rayleigh': solvers[key].medium['rayleigh']}

        # set the generator for the unknown scatterer using ground truth optical properties
        # and unknown extinction.
        # GridToOpticalProperties holds the fixed optical properties and forms a full set of optical properties
        # when it is called with extinction as the argument.
        optical_properties = solvers[wvl].medium['cloud'].copy(deep=True)
        optical_properties = optical_properties.drop_vars('extinction')
        # We are using the ground_truth rte_grid.
        grid_to_optical_properties = at3d.medium.GridToOpticalProperties(
            rte_grid,'cloud', wvl, optical_properties
        )
        # UnknownScatterers is a container for all of the unknown variables.
        # Each unknown_scatterer also records the transforms from the abstract state vector
        # to the gridded data in physical coordinates.
        unknown_scatterers = at3d.containers.UnknownScatterers(
            at3d.medium.UnknownScatterer(grid_to_optical_properties,
            extinction=(None, at3d.transforms.StateToGridMask(mask=cloud_mask))) #carved_volume.mask.data)))
        )
        # now we form state_gen which updates the solvers with an input_state.
        solvers_reconstruct = at3d.containers.SolversDict()
        state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                                 unknown_scatterers, surfaces,
                                                 numerical_parameters, sources, background_optical_scatterers,
                                                 num_stokes)

        # get bounds automatically.
        min_bounds, max_bounds = state_gen.transform_bounds()
        # transform initial physical state to abstract state.
        initial_gridded_extinction = cloud_mask.astype(float)*1.0 #carved_volume.mask.data.astype(float)*1.0
        initial_1d_extinction = state_gen._unknown_scatterers['cloud'].variables['extinction'].state_to_grid.inverse_transform(initial_gridded_extinction)
        x0 = state_gen._unknown_scatterers['cloud'].variables['extinction'].coordinate_transform.inverse_transform(initial_1d_extinction)
        #visualize the initial state
        state_gen(x0)
        forward_sensors.get_measurements(solvers_reconstruct)

        for instrument in forward_sensors:
            for im in forward_sensors.get_images(instrument):
                py.figure()
                im.I.T.plot()


        objective_function = at3d.optimize.ObjectiveFunction.LevisApproxUncorrelatedL2(
            sensors, solvers_reconstruct, forward_sensors, unknown_scatterers, state_gen,
          state_gen.project_gradient_to_state,
            parallel_solve_kwargs={'n_jobs': 4, 'verbose': True},
          gradient_kwargs={'cost_function': 'L2', 'exact_single_scatter':True},
          uncertainty_kwargs={'add_noise': False},
          min_bounds=min_bounds, max_bounds=max_bounds)


        optimizer = at3d.optimize.Optimizer(objective_function)
        warnings.filterwarnings('ignore')
        optimizer._options['maxiter'] = maxiter
        result = optimizer.minimize(x0)

        predicted = solvers_reconstruct[wvl].medium['cloud'].extinction
        truth = solvers[wvl].medium['cloud'].extinction

        # plot radiances
        for instrument in sensors:
            sensor_images = sensors.get_images(instrument)
            fig, ax = py.subplots(1, 9, figsize=(12,4), constrained_layout=True)
            _vmin = []
            _vmax = []
            for i, sensor in enumerate(sensor_images):
                _vmin.append(sensor.I.min())
                _vmax.append(sensor.I.max())
            vmin = np.nanmin(_vmin)
            vmax = np.nanmax(_vmax)
            for i, sensor in enumerate(sensor_images):
                im = ax[i].pcolormesh(sensor.I.data.T, vmin=vmin, vmax=vmax)
                ax[i].axis("off")
                ax[i].set_aspect("equal")
                vza = sensors[instrument]["sensor_list"][i].projection_zenith
                ax[i].set_title(f"{vza:.0f} deg")
            py.colorbar(im, ax=ax[i])
            py.savefig(f"Images_cloud_id{cloud_id:03d}.png", bbox_inches="tight")


    else:
        scene = f"wc_les_RICO_40m_80kmx80km_T_qc_{time:.1f}h"
        _save_dir = os.path.join("cloud_stats", scene)
        if not os.path.exists(_save_dir):
            os.makedirs(_save_dir)
        save_dir = os.path.join(_save_dir, f"cloud_id{cloud_id:03d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # plot true vs predicted volumes
        truth = xr.open_dataset("results/truth_RICO_40m_80kmx80km_T_qc_{time:01.f}h_id{cloud_id:03d}.nc").extinction
        predicted = xr.open_dataset("results/predicted_RICO_40m_80kmx80km_T_qc_{time:0.1f}h_id{cloud_id:03d}.nc").extinction
        error = predicted - truth
        yloc = 22
        r, _ = stats.pearsonr(predicted.data.ravel(), truth.data.ravel())
        slope, offset = np.polyfit(predicted.data.ravel(), truth.data.ravel(), 1)
        cmap = "viridis"
        vlim = np.max(abs(error[:,yloc]))
        vmin = 0
        vmax = np.max([truth[:,yloc].max()])
        mask = truth == 0

        # crop back before plotting
        true_slice, pred_slice, error_slice = fit2canvas(truth[:,yloc].T, predicted[:,yloc].T, error[:,yloc].T, cloud_id)
        mask_slice = true_slice > 0
        # get grid in km
        dx = 0.04
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
        rmse = np.sqrt(np.nanmean(np.ma.masked_array(error_slice, ~mask_slice))**2)
        plt.suptitle(f"Reconstruction cloud {cloud_id}, RMSE={rmse:.1f}"+r" km$^{-1}$")
        ax[0].set_ylabel("z [km]")
        for iax in range(3):
            ax[iax].set_xlabel("x [km]")
            ax[iax].set_aspect("equal")

        plt.savefig(os.path.join(save_dir, f"reconstruction_results_yloc{yloc:03d}_grid.png"))
        true_profile = np.ma.masked_array(true_slice, true_slice==0).mean(axis=1)
        true_profile.data[true_profile.mask] = 0
        pred_profile = np.ma.masked_array(pred_slice, true_slice==0).mean(axis=1)
        pred_profile.data[pred_profile.mask] = 0
        fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), dpi=200, constrained_layout=True)
        rmse_profile = np.sqrt(np.mean((pred_profile.data - true_profile.data)**2))
        ax[0].plot(true_profile.data, xgrid[:,0], label="True", c="k")
        ax[0].plot(pred_profile.data, xgrid[:,0], label="Reconstructed", c="C0")
        ax[0].set_title(f"Vertical Profile, RMSE={rmse_profile:.1f}"+r" km$^{-1}$")
        ax[0].legend()
        ax[0].set_xlabel(r"Extinction [km$^{-1}$]")
        ax[0].set_ylabel("Altitude [km]")
        ax[1] = scatter_density(ax[1], truth.data.ravel(), predicted.data.ravel(), n_sample=len(truth.data.ravel()), x_name="True", y_name="Reconstructed", stratified=True)
        ax[1].plot([0,truth.max()], [0, truth.max()], "k--")
        ax[1].plot([vmin, vmax], [vmin*slope+offset, vmax*slope+offset], "C0--", label=f"linear fit: {slope:.2f}x + {offset:.2f}", zorder=200)
        ax[1].set_xlim(0, truth.max())
        ax[1].set_ylim(0, truth.max())
        ax[1].set_title(fr"R$^2$ = {r:.2f}")
        ax[1].legend()
        ax[1].set_xlabel(r"True Extinction [km$^{-1}$]")
        ax[1].set_ylabel(r"Rec. Extinction [km$^{-1}$]")
        plt.savefig(os.path.join(save_dir, "scatter_profile_results.png"))
        plt.show()
