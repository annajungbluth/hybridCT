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
    if stratified:
        r, _ = stats.pearsonr(_x, _y)  # get R
    else:
        r, _ = stats.pearsonr(x, y)  # get R for subsample
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
    ax.scatter(d_feature, d_target, c=z, s=2, label=r'R$^2$ = ' + str(np.round(r, 2)), zorder=100)
    vmin = np.min([d_feature])#, d_target]) #np.percentile(d_feature,1), np.percentile(d_target,1)])
    vmax = np.max([d_feature])#, d_target]) #np.percentile(d_feature,99), np.percentile(d_target,99)])
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    print(f"slope={slope:.2f}, offset={offset:.2f}")
    _slope, _offset = np.polyfit(x,y, 1)
    print(f"sample slope={_slope:.2f}, offset={_offset:.2f}")
    ax.plot([vmin,vmax],[vmin,vmax], '0.7', ls="--", zorder=200, label="1:1 line")
    y_fit = [vmin*slope+offset, vmax*slope+offset]
    _y_fit = [vmin*_slope+_offset, vmax*_slope+_offset]
    if stratified:
        ax.plot([vmin, vmax], y_fit, "r--", label=f"linear fit: {slope:.2f}x + {offset:.2f}", zorder=200)
    else:
        ax.plot([vmin, vmax], _y_fit, "r--", label=f"linear fit: {_slope:.2f}x + {_offset:.2f}", zorder=200)
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

    time = 10.5
    res = 280 # 40
    tag = "" #"_toa"
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    data_path = "../../data"
    # True optical thickness
    tau_ds = xr.open_dataset(f"{data_path}/training_data/MISR_40m_80x80km_{time:.1f}h_optical_thickness.nc")
    tau = tau_ds.tau.sel(vza=0).data
    _tau_280m = tau_ds.coarsen(x=7, boundary='trim').mean().coarsen(y=7, boundary='trim').mean()
    tau_280m = _tau_280m.tau.sel(vza=0).data
    # Radiance field
    mc_3d = xr.open_dataset(f"{data_path}/training_data/MISR_40m_radiances_phi0_SZA40_RICO_{time:.1f}h_AOD0.1_WIND10.0.nc")
    rad_mc3d = mc_3d.rad.sel(vza=0).data
    rad_mc3d_280m = mc_3d.coarsen(x=7, boundary='trim').mean().coarsen(y=7, boundary='trim').mean().rad.sel(vza=0).data
    rad_mc3d_280m

    # 1D LUT predicted optical thickness - baseline
    lut = np.load("data/lut_disort.npz")
    tau_ret = np.interp(rad_mc3d.ravel(), lut['rad'], lut['tau'], right=np.nan).reshape(rad_mc3d.shape)
    tau_ret_280m = np.interp(rad_mc3d_280m.ravel(), lut['rad'], lut['tau'], right=np.nan).reshape(rad_mc3d_280m.shape)

    sweep = {
            "sza": [20, 30, 50],
            "wind": [5, 10, 20],
            }

    units = {
            "sza": r"$^{\circ}$",
            "wind": " m/s",
            }
    labels = [f'({chr(97 + i)})' for i in range(6)]

    tau_min = 0
    tau_thres = None
    strat = False

    for strat in [True]:
        for tau_thres in [None, 60]:
            counter = 0
            fig, ax = plt.subplots(3, 2, figsize=(8,6), constrained_layout=True)
            for i, param in enumerate(["sza", "wind"]):
                for j, value in enumerate(sweep[param]):
                    print(param, value)
                    # Predicted optical thickness
                    tau_pred_280m = xr.open_dataset(f"{data_path}/predicted_data/MISR_{(res):.0f}m_80x80km_{time:.1f}h_optical_thickness_predicted_ablation_{param}.nc").tau_pre.sel({f"{param}":value}).data

                    if tau_thres is None:
                        cloud_mask = (tau_280m > tau_min) #* (tau_pred_280m > 2)
                    else:
                        cloud_mask = (tau_280m > tau_min) * (tau_280m <= tau_thres+0.01) #* (tau_pred_280m > 2)
                    scatter_density(ax[j,i], tau_280m[cloud_mask], tau_pred_280m[cloud_mask], "True", "Predicted", title="", stratified=strat)
                    ax[j,i].text(-0.3, 1.05, labels[counter], transform=ax[j,i].transAxes, fontsize='medium', fontweight='bold', va='bottom')
                    ax[j,1].set_ylabel("")
                    ax[j,i].set_xlabel("")
                    ax[j,i].set_title(f"An, {param.upper()}={value:.1f}"+units[param])#+r"$^{\circ}$")
                    counter+=1
                ax[-1,i].set_xlabel("True")
            if tau_thres is None:
                plt.savefig(f"{plot_dir}/tau_pred_unet_scatter_ablation_stratified{strat}_{time:.1f}h_280m.png")
            else:
                plt.savefig(f"{plot_dir}/tau_pred_unet_scatter_ablation_stratified{strat}_<{tau_thres}_{time:.1f}h_280m.png")
            plt.show()

    # Reference plots to compare default nadir performance
    #tau_pred_280m = xr.open_dataset(f"{data_path}/predicted_data/MISR_280m_80x80km_{time:.1f}h_optical_thickness_predicted{tag}.nc").tau_pre.sel(vza=0)
    param = "wind"
    value = 10
    tau_pred_280m = xr.open_dataset(f"{data_path}/predicted_data/MISR_{(res):.0f}m_80x80km_{time:.1f}h_optical_thickness_predicted_ablation_{param}.nc").tau_pre.sel({f"{param}":value})
    cloud_mask = tau_280m > tau_min
    #nan_mask = tau_pred[cloud_mask] == tau_pred[cloud_mask]
    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax = scatter_density(ax, tau_280m[cloud_mask], tau_pred_280m.data[cloud_mask], "True", "Predicted", title="U-Net", stratified=True)#, n_sample=len(tau_ret[cloud_mask][nan_mask])) #, y_name
    plt.savefig(f"{plot_dir}/tau_pred_unet_scatter_ablation_{time:.1f}h_280m.png")
    plt.show()

    thres = 60
    cloud_mask = (tau_280m <= thres+0.01) & (tau_280m > tau_min)
    nan_mask = tau_ret_280m[cloud_mask] == tau_ret_280m[cloud_mask]
    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax = scatter_density(ax, tau_280m[cloud_mask][nan_mask], tau_ret_280m[cloud_mask][nan_mask], "True", "Predicted", title="1D LUT", stratified=True)
    plt.savefig(f"{plot_dir}/tau_pred_lut_scatter_ablation_<{thres}_{time:.1f}h_280m.png")
    plt.show()

    fig = plt.figure(figsize=(5,2.5), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax = scatter_density(ax, tau_280m[cloud_mask], tau_pred_280m.data[cloud_mask], "True", "Predicted", title="U-Net", stratified=True)
    plt.savefig(f"{plot_dir}/tau_pred_unet_scatter_ablation_<{thres}_{time:.1f}h_280m.png")
    plt.show()
