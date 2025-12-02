import numpy as np
import os
from scipy import stats
import xarray as xr
import pytomo.mystic.generate_cloud_file as gencld
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import random

plt.ion()
cmap = mpl.colormaps.get_cmap('viridis').copy()
cmap.set_under('w')
mpl.rcParams['figure.dpi'] = 200
my_cmap = mpl.colormaps.get_cmap('viridis').copy()
my_cmap.set_under('w')
my_bwr = mpl.colormaps.get_cmap('bwr').copy()
my_bwr.set_under('w')

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

def scatter_density(ax, _x, _y, x_name, y_name, title, save_dir=None, n_sample=10000, stratified=False):
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
    #fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    #ax = fig.add_subplot(111)
    ax.grid(ls=":", zorder=-100)
    ax.scatter(_x, _y, c="0.9", s=2)
    ax.scatter(d_feature, d_target, c=z, s=2, label='R = ' + str(np.round(r, 2)), zorder=100)
    vmin = np.min([d_feature])#, d_target]) #np.percentile(d_feature,1), np.percentile(d_target,1)])
    vmax = np.max([d_feature])#, d_target]) #np.percentile(d_feature,99), np.percentile(d_target,99)])
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    print(f"slope={slope:.2f}, offset={offset:.2f}")
    _slope, _offset = np.polyfit(x,y, 1)
    print(f"sample slope={_slope:.2f}, offset={_offset:.2f}")
    ax.plot([vmin,vmax],[vmin,vmax], '0.7', ls="--", zorder=200, label="1:1 line")
    y_fit = [vmin*slope+offset, vmax*slope+offset]
    ax.plot([vmin, vmax], y_fit, "r--", label=f"linear fit: {slope:.2f}x + {offset:.2f}", zorder=200)
    # constrained
    #slope_constrained = np.sum(x * y) / np.sum(x ** 2)
    #y_fit = [vmin*slope_constrained, vmax*slope_constrained]
    #ax.plot([vmin, vmax], y_fit, "m--", label=f"linear fit: {slope_constrained:.2f}x", zorder=200)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    #plt.tight_layout()
    return ax

if __name__ == "__main__":

    time = 10.5 #30
    res = 280 # 40
    vza = 60
    tag = "" #"_toa"
    data_path = "/Users/lforster/Documents/deep_learning/cloud_tomography"
    data_path = "/Volumes/Fortress_L3/RICO_40m_diurnal_cycle"
    lut = np.load("/Users/lforster/Documents/deep_learning/cloud_tomography/rt_simulations/RICO_40m_diurnal_cycle/IPA_DISORT/lut_disort.npz")
    tau_ds = xr.open_dataset(f"{data_path}/training_data/MISR_40m_80x80km_{time:.1f}h_optical_thickness.nc")
    #tau_pred_280m = xr.open_dataset(f"{data_path}/predicted_data/MISR_280m_80x80km_{time:.1f}h_optical_thickness_predicted{tag}.nc").tau_pre.sel(vza=0)
    tau_pred_280m = xr.open_dataset(f"{data_path}/predicted_data/MISR_{res}m_80x80km_{time:.1f}h_optical_thickness_predicted_MIMO_toa.nc").tau_pre.sel(vza=vza)
    #tau_pred_280m = tau_pred.coarsen(x=7, boundary='trim').mean().coarsen(y=7, boundary='trim').mean().data
    tau = tau_ds.tau.sel(vza=vza).data
    _tau_280m = tau_ds.coarsen(x=7, boundary='trim').mean().coarsen(y=7, boundary='trim').mean()
    tau_280m = _tau_280m.tau.sel(vza=vza).data
    #mc_3d = xr.open_dataset("/Volumes/Fortress_L3/RICO_40m_diurnal_cycle/RICO_30.0h_AOD0.0_ALBEDO0.0/sza0/results/MISR_40m_radiances_phi0_SZA0_RICO_30.0h_AOD0.0_ALBEDO0.0.nc")
    mc_3d = xr.open_dataset(f"{data_path}/training_data/MISR_40m_radiances_phi0_SZA40_RICO_{time:.1f}h_AOD0.1_WIND10.0.nc")
    rad_mc3d = mc_3d.rad.sel(vza=vza).data
    rad_mc3d_280m = mc_3d.coarsen(x=7, boundary='trim').mean().coarsen(y=7, boundary='trim').mean().rad.sel(vza=vza).data
    rad_mc3d_280m

    tau_ret = np.interp(rad_mc3d.ravel(), lut['rad'], lut['tau'], right=np.nan).reshape(rad_mc3d.shape)
    tau_ret_280m = np.interp(rad_mc3d_280m.ravel(), lut['rad'], lut['tau'], right=np.nan).reshape(rad_mc3d_280m.shape)

    vmax = rad_mc3d_280m.max()
    vmin = rad_mc3d_280m.min()
    cloud_mask_280m = tau_280m>0
    x_grid, y_grid = np.meshgrid(_tau_280m.x.data, _tau_280m.y.data)
    fig, ax = plt.subplots(1,2, figsize=(7.5,3), sharey=False, constrained_layout=True)
    im = ax[0].pcolormesh(x_grid, y_grid, np.ma.masked_array(rad_mc3d_280m, ~cloud_mask_280m), cmap=my_cmap)#, norm=mcolors.PowerNorm(0.3, vmin=vmin, vmax=vmax))
    ax[0].set_title('3D MYSTIC')
    cbar = plt.colorbar(im, ax=ax[0])
    cbar.set_label('Radiance [mW/(m2 nm sr)]')
    diff = np.ma.masked_array(tau_ret_280m - tau_280m, ~cloud_mask_280m)
    #diff = rad_mc_ipa - mc_3d['rad']
    #diff_im = ax[1].pcolormesh(np.ma.masked_array(diff, cloud_mask), vmin=-vlim, vmax=vlim, cmap=my_bwr)
    diff_im = ax[1].pcolormesh(x_grid, y_grid, diff, cmap="coolwarm", vmin=-50, vmax=50) #my_cmap, norm=mcolors.PowerNorm(0.3, vmin=0, vmax=350))
    #ax[1].contour(~cloud_mask, levels=[0], colors="k")
    cbar = plt.colorbar(diff_im, ax=ax[1])
    cbar.set_label('Radiance [mW/(m2 nm sr)]')
    ax[1].set_title('Difference\n3D MYSTIC - 1D DISORT')
    ax[0].set_ylabel('y [km]')
    for i in range(2):
        ax[i].set_xlabel('x [km]')
        ax[i].set_aspect('equal')
    #plt.tight_layout()
    plt.savefig(f"rad_3D_1D_diff_{time:.1f}h_mimo_280m.png")
    plt.show()

    vmax = tau_280m.max()
    vmin = tau_280m.min()
    cloud_mask_280m = tau_280m>0
    fig, ax = plt.subplots(1, 3, figsize=(10,3), sharey=False, dpi=150, constrained_layout=True)
    im = ax[0].pcolormesh(x_grid, y_grid, np.ma.masked_array(tau_280m, ~cloud_mask_280m), cmap=my_cmap, norm=mcolors.PowerNorm(0.3, vmin=vmin, vmax=vmax))
    ax[0].set_title('True Optical Thickness')
    cbar = plt.colorbar(im, ax=ax[0])
    cbar.set_label('Optical Thickness []')
    ax[1].set_title("MIMO U-Net\n"+r'Difference: True - Predicted')
    diff = np.ma.masked_array(tau_pred_280m - tau_280m, ~cloud_mask_280m)
    diff_im = ax[1].pcolormesh(x_grid, y_grid, diff, cmap="coolwarm", vmin=-50, vmax=50) #my_cmap, norm=mcolors.PowerNorm(0.3, vmin=0, vmax=350))
    #ax[1].contour(~cloud_mask, levels=[0], colors="k")
    cbar = plt.colorbar(diff_im, ax=ax[2])
    cbar.set_label('Optical Thickness Diff. []')
    ax[2].set_title("1D LUT\n"+r"Difference: True - Predicted")
    diff = np.ma.masked_array(tau_ret_280m - tau_280m, ~cloud_mask_280m)
    #diff = rad_mc_ipa - mc_3d['rad']
    #diff_im = ax[1].pcolormesh(np.ma.masked_array(diff, cloud_mask), vmin=-vlim, vmax=vlim, cmap=my_bwr)
    diff_im = ax[2].pcolormesh(x_grid, y_grid, diff, cmap="coolwarm", vmin=-50, vmax=50) #my_cmap, norm=mcolors.PowerNorm(0.3, vmin=0, vmax=350))
    cbar = plt.colorbar(diff_im, ax=ax[1])
    cbar.set_label('Optical Thickness Diff. []')
    ax[0].set_ylabel('y [km]')
    for i in range(3):
        ax[i].set_xlabel('x [km]')
        ax[i].set_aspect('equal')
    #plt.tight_layout()
    plt.savefig(f"tau_3D_1D_diff_{time:.1f}h_mimo_280m.png")
    plt.show()

    tau_min = 0
    cloud_mask = (tau_280m > tau_min) * (tau_ret_280m > 2)#tau_min)
    nan_mask = tau_ret_280m[cloud_mask] == tau_ret_280m[cloud_mask]
    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    scatter_density(ax, tau_280m[cloud_mask][nan_mask], tau_ret_280m[cloud_mask][nan_mask], "True", "Predicted", title="1D LUT", stratified=True)#, n_sample=len(tau_ret[cloud_mask][nan_mask])) #, y_name
    plt.savefig(f"tau_pred_lut_scatter_{time:.1f}h_mimo_280m.png")
    plt.show()

    #cloud_mask = tau_280m > tau_min
    cloud_mask = (tau_280m > tau_min) * (tau_pred_280m > 2)#tau_min)
    #nan_mask = tau_pred[cloud_mask] == tau_pred[cloud_mask]
    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    scatter_density(ax, tau_280m[cloud_mask], tau_pred_280m.data[cloud_mask], "True", "Predicted", title="MIMO U-Net", stratified=True)#, n_sample=len(tau_ret[cloud_mask][nan_mask])) #, y_name
    plt.savefig(f"tau_pred_unet_scatter_{time:.1f}h_mimo_280m.png")
    plt.show()

    thres = 60
    cloud_mask = (tau_280m <= thres+0.01) & cloud_mask #(tau_280m > tau_min)
    nan_mask = tau_ret_280m[cloud_mask] == tau_ret_280m[cloud_mask]
    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    scatter_density(ax, tau_280m[cloud_mask][nan_mask], tau_ret_280m[cloud_mask][nan_mask], "True", "Predicted", title="1D LUT", stratified=True)
    plt.savefig(f"tau_pred_lut_scatter_<{thres}_{time:.1f}h_mimo_280m.png")
    plt.show()

    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    scatter_density(ax, tau_280m[cloud_mask], tau_pred_280m.data[cloud_mask], "True", "Predicted", title="MIMO U-Net", stratified=True)
    plt.savefig(f"tau_pred_unet_scatter_<{thres}_{time:.1f}h_mimo_280m.png")
    plt.show()


    tau_pred_280m_ds = xr.open_dataset(f"{data_path}/predicted_data/MISR_{res}m_80x80km_{time:.1f}h_optical_thickness_predicted_MIMO_toa.nc")
    vza_directions = {
            "aft": np.array([26.1, 45.6, 60.0, 70.5]),
            "fwd": -np.array([26.1, 45.6, 60.0, 70.5]),
            }
    fig, ax = plt.subplots(4, 2, figsize=(8,6), constrained_layout=True)
    for i, direction in enumerate(["aft", "fwd"]):
        for j, vza in enumerate(vza_directions[direction]):
            tau_pred_280m = tau_pred_280m_ds.tau_pre.sel(vza=vza).data
            tau_280m = tau_ds.tau.sel(vza=vza).coarsen(x=7, boundary='trim').mean().coarsen(y=7, boundary='trim').mean().data
            cloud_mask = (tau_280m > tau_min) * (tau_pred_280m > 2)
            scatter_density(ax[j,i], tau_280m[cloud_mask], tau_pred_280m[cloud_mask], "True", "Predicted", title="", stratified=True)
    plt.show()
