import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import stats
import numpy as np
import os
from emulator.mimo.pyct.evaluation import tiled_rmse, difference_map, tiled_difference_map, tiled_rmse_cloud_only
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xarray as xr

def heatmap(predicted, truth, title="Optical Thickness", save_dir=None, nbins=100, cmap="viridis", ticks=[1e-6,1e-3,0.1,1,10,100]):
    r, _ = stats.pearsonr(predicted, truth)
    print("correlation coeff. heatmap:", r)
    # exclude zeros from log
    zero_mask = (predicted > 0) * (truth > 0)
    # compute 2D histogram
    h, xedges, yedges = np.histogram2d(np.log(truth[zero_mask]), np.log(predicted[zero_mask]), bins=nbins)
    # set 0 to white
    my_cmap = mpl.cm.get_cmap(cmap).copy()
    my_cmap.set_under('w')

    fig, ax = plt.subplots(1, 1, dpi=120)
    im = ax.pcolormesh(xedges, yedges, np.ma.masked_array(h.T, mask=(h.T==0)), cmap=my_cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts')
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Truth")
    xmin = np.log(ticks[0])
    xmax = xedges.max()*1.1
    ymin = np.log(ticks[0])
    ymax = yedges.max()*1.1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{title}, R = {r:.2f}")
    # add 1:1 line
    pmin = np.min([xmin, ymin])
    pmax = np.max([xmax, ymax])
    ax.plot([pmin,pmax],[pmin,pmax], '0.8', ls="--", label="R = 1.0")
    ax.legend()
    # set ticks and ticklabels
    ax.set_xticks(np.log(ticks))
    ax.set_yticks(np.log(ticks))
    ticklabels = [f"{t:.0e}" for t in ticks]
    ax.set_xticklabels(ticklabels, rotation=45)
    ax.set_yticklabels(ticklabels)
    # set grid
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title.lower().replace(" ","_").replace(".","")))
    plt.close()

def plot_nLowest_nHighest(X, y, y_hat, cloud_fraction_thres=0.5, n=3, save_dir=None):
    cloud_fraction = np.mean(y>0, axis=(1,2))
    rmse_tiles = np.sqrt(y - y_hat)**2
    print(f"filter RMSE for {cloud_fraction_thres} or more cloud fraction.")
    rmse_tiles_cloudy = rmse_tiles[cloud_fraction >= cloud_fraction_thres]
    indices = np.argsort(np.nanmean(rmse_tiles_cloudy, axis=(1,2)))
    #print(indices)
    if len(indices < n):
        return
    rmse_idx = {
        "lowest": indices[:n],
        "highest": indices[-n:]
        }
    # color limits
    xmin = X.min()
    xmax = X.max()
    ymin = y.min()
    ymax = y.max()
    vlim = np.max([abs(ymax), abs(ymin)])

    for plot_type in ["highest", "lowest"]:
        fig, ax = plt.subplots(n, 4, figsize=(8,4), sharey=True, sharex=True, dpi=120)
        for iax in range(n):
            idx = rmse_idx[plot_type][iax]
            ax[iax,0].set_ylabel(f"Tile index {idx}\nRMSE={np.nanmean(rmse_tiles_cloudy[idx]):.3f}")
            im1=ax[iax,0].pcolormesh(np.rot90(X[idx]))
            ymin = np.min([y[idx].min(), y_hat[idx].min()])
            ymax = np.max([y[idx].max(), y_hat[idx].max()])
            vlim = np.max(abs(y[idx]-y_hat[idx]))
            im2=ax[iax,1].pcolormesh(np.rot90(y[idx]), vmin=ymin, vmax=ymax)
            im3=ax[iax,2].pcolormesh(np.rot90(y_hat[idx]), vmin=ymin, vmax=ymax)
            im4=ax[iax,3].pcolormesh(np.rot90(y[idx] - y_hat[idx]), cmap="bwr", vmin=-vlim, vmax=vlim)
            plt.colorbar(im1,ax=ax[iax,0])
            plt.colorbar(im2,ax=ax[iax,1])
            plt.colorbar(im3,ax=ax[iax,2])
            plt.colorbar(im4,ax=ax[iax,3])
        ax[0,0].set_title(f"Rad")
        ax[0,1].set_title(f"Tau")
        ax[0,2].set_title(f"Tau pred.")
        ax[0,3].set_title(f"Tau pred.-true")
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"tiles_with_{plot_type}_rmse.png"))
            plt.close()


def panel_plot(X_map, y_map, y_hat_map, angles, apply_mask=True, color_limits=True, alea_uncert=None, epis_uncert=None):
    '''
    Plot panel with 4 columns for X_map, y_map, y_hat_map, and y_map - y_hat_map.
    Will expand into multiple rows for `len(angles) > 1`.
    :param X_map: input data (shape [n_angles,n_x,n_y])
    :param y_map: target data (shape [n_angles,n_x,n_y])
    :param y_hat_map: predicted data (shape [n_angles,n_x,n_y])
    :param angles: list of viewing angles (shape [n_angles])
    :return: fig
    '''

    if color_limits:
        xmin = X_map.min()
        xmax = X_map.max()
        ymin = y_map.min()
        ymax = y_map.max()
        vmax = np.max([abs(ymax), abs(ymin)])
        vmin = -vmax
        uncert_min = 0.0
        uncert_max = 1.0
    else:
        vmin = vmax = xmin = xmax = ymin = ymax = None
    nrows = 4
    if alea_uncert is not None:
        nrows += 2
    ncols = len(angles)
    height = 8
    width = 7/3.
    if angles and len(angles) > 1:
        width *= ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(width,height), sharey=True, sharex=True, dpi=200)
    for ivza, vza in enumerate(angles):
        if apply_mask:
            mask = y_map[...,ivza] == 0
        else:
            mask = np.zeros_like(y_map[...,ivza], dtype="bool")
        # handle both 1D and 2D panels
        if ncols > 1:
            _ax = ax[:,ivza]
        else:
            _ax = ax
        im5 = None
        im6 = None
        im1 = _ax[0].pcolormesh(np.flipud(np.rot90(np.ma.masked_array(X_map[...,ivza], mask))), vmin=xmin, vmax=xmax)
        im2 = _ax[1].pcolormesh(np.flipud(np.rot90(np.ma.masked_array(y_map[...,ivza], mask))), vmin=ymin, vmax=ymax)
        im3 = _ax[2].pcolormesh(np.flipud(np.rot90(np.ma.masked_array(y_hat_map[...,ivza], mask))), vmin=ymin, vmax=ymax)
        im4 = _ax[3].pcolormesh(np.flipud(np.rot90(np.ma.masked_array(y_hat_map[...,ivza] - y_map[...,ivza], mask))), cmap="bwr", vmin=vmin, vmax=vmax)
        if alea_uncert is not None and epis_uncert is not None:
            #print(X_map.shape, alea_uncert.shape, epis_uncert.shape)
            im5 = _ax[4].pcolormesh(np.flipud(np.rot90(np.ma.masked_array(alea_uncert[...,ivza], mask))), vmin=uncert_min, vmax=uncert_max)
            im6 = _ax[5].pcolormesh(np.flipud(np.rot90(np.ma.masked_array(epis_uncert[...,ivza], mask))), vmin=uncert_min, vmax=uncert_max)

        for k in range(nrows):
            _ax[k].set_aspect('equal')
        _ax[0].set_title(f"VZA={angles[ivza]}")
    # handle both 1D and 2D panels
    if ncols > 1:
        ax0 = ax[:,0]
        axLast = ax[:,-1]
    else:
        ax0 = ax
        axLast = ax
    ax0[0].set_ylabel(f"Rad")
    ax0[1].set_ylabel(f"Tau")
    ax0[2].set_ylabel(f"Tau pred.")
    ax0[3].set_ylabel(f"Tau pred.-true")
    cbars = [im1, im2, im3, im4]
    if alea_uncert is not None:
        cbars.extend([im5, im6])
        ax0[4].set_ylabel(f"Aleat. uncert.")
        ax0[5].set_ylabel(f"Epist. uncert.")
    cb = []
    for iax, _ax in enumerate(axLast):
        divider = make_axes_locatable(_ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = plt.colorbar(cbars[iax], cax=cax)
        cbar.set_label("[]")
        cb.append(cbar)
    cb[0].set_label(r"[mW/(m$^2$ nm sr)]")
    return fig


#Can be used for full image or tiles
def gen_plots(y_hat, y, X, uid, ymin=-10,ymax=10,xmin=0,xmax=10, out_dir=".", log_scale = True,
        plot_metrics = True, cloud_threshold=np.inf, tile_size = 10, angles=[], alea_uncert = None, epis_uncert = None):

    #if y.ndim > 3:
    #    y = np.squeeze(y)
    #    y_hat = np.squeeze(y_hat)


    n_samples = y.shape[0]
    for j in range(n_samples):
        # plot 4 x n_views panels for X, y, y_hat, and y_hat - y maps
        if alea_uncert is not None and epis_uncert is not None:
            fig = panel_plot(X[j], y[j], y_hat[j], angles=angles, alea_uncert=alea_uncert[j], epis_uncert=epis_uncert[j])
        else:
            fig = panel_plot(X[j], y[j], y_hat[j], angles=angles)
        plt.savefig(os.path.join(out_dir,"result_" + str(uid) + ".png"), bbox_inches='tight')
        plt.close()
        if plot_metrics:
            if angles and len(angles) > 1:
                # select nadir view
                nadir_idx = int(y[j].shape[2] / 2) #np.where(np.array(angles)==0)[0][0]
                tmp1 = y_hat[j]
                tmp2 = y[j]
                tmp3 = X[j]
                t_rmse = tiled_rmse(tmp1[:,:,nadir_idx:nadir_idx+1], \
                        tmp2[:,:,nadir_idx:nadir_idx+1], tile_size)
                t_diff_map = tiled_difference_map(tmp1[:,:,nadir_idx:nadir_idx+1], \
                        tmp2[:,:,nadir_idx:nadir_idx+1], tile_size)
                diff_map = difference_map(tmp1[:,:,nadir_idx:nadir_idx+1], tmp2[:,:,nadir_idx:nadir_idx+1])
                #print("3 best - 3 worst", tmp3.shape, tmp2.shape, tmp1.shape)
                #plot_nLowest_nHighest(tmp3[:,:,nadir_idx:nadir_idx+1], tmp2[:,:,nadir_idx:nadir_idx+1], tmp1[:,:,nadir_idx:nadir_idx+1], save_dir=out_dir)
            else:
                t_rmse = tiled_rmse(y_hat[j,:,:], y[j,:,:], tile_size)
                t_diff_map = tiled_difference_map(y_hat[j,:,:], y[j,:,:], tile_size)
                diff_map = difference_map(y_hat[j,:,:], y[j,:,:])
                #print("3 best - 3 worst")
                #plot_nLowest_nHighest(X[j,:,:], y[j,:,:], y_hat[j,:,:], save_dir=out_dir)
 
            
            # plot difference optical thickness
            RMSE = np.sqrt(np.mean((y_hat[j,:,:]-y[j,:,:])**2))
            RMSE_rel = (RMSE / y[j].max()) * 100
            diff = xr.DataArray(np.squeeze(diff_map), dims=["x","y"])
            if angles and len(angles) > 1:
                tau = xr.DataArray(np.squeeze(y[j,:,:,nadir_idx:nadir_idx+1]), dims=["x","y"])
            else:
                tau = xr.DataArray(np.squeeze(y[j,:,:]), dims=["x","y"])
            fig, ax = plt.subplots(1,1, figsize=(3.5,2.5), constrained_layout=True)
            im = diff.plot(x="x", y="y", cmap="bwr", vmin=-50, vmax=50)
            tau.plot.contour(x="x", y="y", levels=[0.5], colors='k', linewidths=0.2)
            ax.set_title(f"RMSE = {RMSE:.1f}, rel. RMSE = {RMSE_rel:.1f} %")
            plt.savefig(os.path.join(out_dir, f"diff_vza{angles[0]:.1f}.png"))
            plt.close()

            plt.figure()
            plt.imshow(np.squeeze(t_rmse))
            plt.colorbar()
            plt.savefig(os.path.join(out_dir,"tiled_rmse_" + str(uid) + ".png"), bbox_inches='tight')
            plt.close()

            #plt.imshow(np.squeeze(diff_map))
            #plt.colorbar()
            #plt.savefig(os.path.join(out_dir,"difference_map_" + str(uid) + ".png"), bbox_inches='tight')
            #plt.clf()


            plt.imshow(np.squeeze(t_diff_map))
            plt.colorbar()
            plt.savefig(os.path.join(out_dir,"tiled_difference_map_" + str(uid) + ".png"), bbox_inches='tight')
            plt.clf()


            if np.isfinite(cloud_threshold):
                if angles and len(angles) > 1:
                    # select nadir view
                    nadir_idx = int(y[j].shape[2] / 2) #np.where(np.array(angles)==0)[0][0]
                    tmp1 = y_hat[j]
                    tmp2 = y[j]
                    tmp3 = X[j]
                    t_rmse = tiled_rmse(tmp1[:,:,nadir_idx:nadir_idx+1], \
                        tmp2[:,:,nadir_idx:nadir_idx+1], tile_size)
                    t_diff_map = tiled_difference_map(tmp1[:,:,nadir_idx:nadir_idx+1], \
                        tmp2[:,:,nadir_idx:nadir_idx+1], tile_size)
                    diff_map = difference_map(tmp1[:,:,nadir_idx:nadir_idx+1], tmp2[:,:,nadir_idx:nadir_idx+1])
                    #print("3 best - 3 worst")
                    #plot_nLowest_nHighest(tmp3[:,:,nadir_idx:nadir_idx+1], tmp2[:,:,nadir_idx:nadir_idx+1], tmp1[:,:,nadir_idx:nadir_idx+1], save_dir=out_dir)
                else:
                    t_rmse = tiled_rmse(y_hat[j,:,:], y[j,:,:], tile_size)
                    t_diff_map = tiled_difference_map(y_hat[j,:,:], y[j,:,:], tile_size)
                    diff_map = difference_map(y_hat[j,:,:], y[j,:,:])
                    #print("3 best - 3 worst")
                    #plot_nLowest_nHighest(X[j,:,:], y[j,:,:], y_hat[j,:,:], save_dir=out_dir)







                plt.imshow(np.squeeze(t_rmse))
                plt.colorbar()
                plt.savefig(os.path.join(out_dir,"tiled_rmse_cloud_only_" + str(uid) + ".png"), bbox_inches='tight')
                plt.clf()

                # plot difference optical thickness
                RMSE = np.sqrt(np.mean((y_hat[j,:,:]-y[j,:,:])**2))
                RMSE_rel = (RMSE / y[j].max()) * 100 
                diff = xr.DataArray(np.squeeze(diff_map), dims=["x","y"])
                if angles and len(angles) > 1:
                    tau = xr.DataArray(np.squeeze(y[j,:,:,nadir_idx:nadir_idx+1]), dims=["x","y"])
                else:
                    tau = xr.DataArray(np.squeeze(y[j,:,:]), dims=["x","y"])
                fig, ax = plt.subplots(1,1, figsize=(3.5,2.5), constrained_layout=True)
                im = diff.plot(x="x", y="y", cmap="bwr", vmin=-50, vmax=50)
                tau.plot.contour(x="x", y="y", levels=[0.5], colors='k', linewidths=0.2)
                ax.set_title(f"RMSE = {RMSE:.1f}, rel. RMSE = {RMSE_rel:.1f} %")
                plt.savefig(os.path.join(out_dir, f"diff_cloud_only_vza{angles[0]:.1f}.png"))
                plt.close()
                #plt.imshow(np.squeeze(diff_map))
                #plt.colorbar()
                #plt.savefig(os.path.join(out_dir,"difference_map_cloud_only_" + str(uid) + ".png"), bbox_inches='tight')
                #plt.clf()

                plt.imshow(np.squeeze(t_diff_map))
                plt.colorbar()
                plt.savefig(os.path.join(out_dir,"tiled_difference_map_cloud_only_" + str(uid) + ".png"), bbox_inches='tight')
                plt.clf()

                title = f"Optical Thickness"
                scatter_density(y.ravel(), y_hat.ravel(), "True", "Predicted", title, out_dir)
                # density plot for COT < 60
                cot_thres = 60
                scatter_density(y[y<cot_thres], y_hat[y<cot_thres], "True", "Predicted", title+f" COT<{cot_thres}", out_dir)
                if alea_uncert is not None:
                    # un-scale uncertainty
                    alea_unscale = MinMaxScaler().fit_transform(alea_uncert.reshape(-1,1)).ravel()
                    epis_unscale = MinMaxScaler().fit_transform(epis_uncert.reshape(-1,1)).ravel()
                    scatter_uncert(y.ravel(), y_hat.ravel(), alea_unscale.ravel(), "True", "Predicted", "Aleatoric Uncertainty", out_dir)
                    scatter_uncert(y.ravel(), y_hat.ravel(), epis_unscale.ravel(), "True", "Predicted", "Epistemic Uncertainty", out_dir)
                    plt.figure()
                    plt.scatter(y.ravel(), alea_unscale.ravel(), color="b")
                    plt.xlabel("True Optical Thickness")
                    plt.ylabel("Aleatoric Uncertainty")
                    plt.savefig(os.path.join(out_dir, "scatter_true_vs_aleatoric.png"))
                    plt.figure()
                    plt.scatter(y.ravel(), epis_unscale.ravel(), color="b")
                    plt.xlabel("True Optical Thickness")
                    plt.ylabel("Epistemic Uncertainty")
                    plt.savefig(os.path.join(out_dir, "scatter_true_vs_epistemic.png"))

                if np.isfinite(cloud_threshold):
                    title = "Optical Thickness Cloud Only"
                    inds = np.where(y >= cloud_threshold)
                    scatter_density(y[inds].ravel(), y_hat[inds].ravel(), "True", "Predicted", title, out_dir)
 
                print("heatmap plot")
                # heatmap for all data
                title = "Optical Thickness "
                heatmap(y.ravel(), y_hat.ravel(), title=title, save_dir=out_dir)

        if angles and len(angles) > 1:
            for ivza, vza in enumerate(angles):
                title = f"Optical Thickness VZA={vza}deg"
                heatmap(y_hat[:,:,ivza:ivza+1].ravel(), y[:,:,ivza:ivza+1].ravel(), title=title, save_dir=out_dir)

        if isinstance(uid, int):
            uid += 1
        else:
            uid += "_" + str(j)


def scatter_uncert(x, y, uncert, x_name, y_name, title, save_dir=None, n_sample=5000):
    '''
    makes a scatter plot and color codes where most of the data is
    :param x: x-value
    :param y: y-value
    :param uncert: uncertainties
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param save_dir: save location
    :return: -
    '''
    print("HERE UQ", uncert.min(), uncert.max())
    fig = plt.figure(figsize=(4,3), dpi=150, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.grid(ls=":", zorder=-100)
    # ensure higher uncertainties are plotted last/on top
    sort_idx = np.argsort(x)
    im = ax.scatter(x[sort_idx], y[sort_idx], c=uncert[sort_idx], s=2, zorder=100, cmap="coolwarm")#, alpha=0.5)
    #im = ax.scatter(x[uncert >= 0.5], y[uncert >= 0.5], c=uncert[uncert >= 0.5], s=2, zorder=100, cmap="bwr", alpha=0.5)
    #im.set_clim(0.0,1)
    fig.colorbar(im, ax=ax)
    #ax.scatter(x[uncert >= 0.5], y[uncert >= 0.5], c=uncert[uncert >= 0.5], s=2, zorder=100, cmap="bwr", alpha=0.5)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(x))
    if save_dir is not None:
        save_fname = os.path.join(save_dir, title.lower().replace(" ","_")+"_scatter_density.png")
        fig.savefig(save_fname)
        plt.close()
 
def calculate_density(x, y, n_sample):
    # select samples
    if n_sample > len(x) or n_sample < 1:
        n_sample = len(x)
    rand_idx = np.random.randint(0, len(x)-1, n_sample)
    x = x[rand_idx]
    y = y[rand_idx]
    xy = np.vstack([x, y])
    if np.mean(x) == np.mean(y):
        z = np.arange(len(x))
    else:
        try:
            z = stats.gaussian_kde(xy)(xy)  # calculate density
        except:
            return
    # sort points by density
    idx = z.argsort()
    d_feature = x[idx]
    d_target = y[idx]
    z = z[idx]
    return d_feature, d_target, z


def scatter_density(x, y, x_name, y_name, title, save_dir=None, n_sample=-1):
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
    #max_n = 50000
    try:
        r, _ = stats.pearsonr(x, y)  # get R
        print("correlation coeff. scatter density:", r)
    except:
        print('could not calculate r, set to nan')
        r = np.nan
    # linear fit
    slope, offset = np.polyfit(x,y, 1)
    # get density for random subset of data
    d_feature, d_target, z = calculate_density(x, y, n_sample)
    # plot everything
    fig = plt.figure(figsize=(5,2.5), dpi=120, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.grid(ls=":", zorder=-100)
    ax.scatter(x, y, c="0.9")
    if r is not np.nan:
        ax.scatter(d_feature, d_target, c=z, s=2, label='R = ' + str(np.round(r, 2)), zorder=100)
    else:
        ax.scatter(d_feature, d_target, c=z, s=2, zorder=100)
    vmin = min(min(x), min(y))
    vmax = max(max(x), max(y))
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.plot([vmin,vmax],[vmin,vmax], '0.7', ls="--", zorder=200, label="1:1 line")
    ax.plot([vmin, vmax], [vmin*slope+offset, vmax*slope+offset], "r--", label=f"linear fit: {slope:.2f}x + {offset:.2f}", zorder=200)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)

    if save_dir is not None:
        save_fname = os.path.join(save_dir, title.lower().replace(" ","_")+"_scatter_density.png")
        fig.savefig(save_fname)
        plt.close()

