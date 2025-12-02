from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

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
