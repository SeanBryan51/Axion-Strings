import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import subprocess, os, glob

from scipy.fftpack import fftn, fftfreq, fftshift
from scipy.stats import binned_statistic

# initial thermal spectrum

def pkphi(k, T_initial=4.0, lambdaPRS=1.0, scalefactor=1.0):
    T = T_initial / scalefactor
    m_eff_squared = lambdaPRS * (T**2 / 3 - 1)
    omega = np.sqrt(k**2 + scalefactor**2*m_eff_squared);
    bose = 1 / (np.exp(omega / T) - 1);
    return bose / omega

def pkphidot(k, T_initial=4.0, lambdaPRS=1.0, scalefactor=1.0):
    T = T_initial / scalefactor
    m_eff_squared = lambdaPRS * (T**2 / 3 - 1)
    omega = np.sqrt(k**2 + scalefactor**2*m_eff_squared);
    bose = 1 / (np.exp(omega / T) - 1);
    return bose * omega


# power spectrum estimator:

def power_spectrum2D(f,N,L):
    y = fftn(f)
    P = np.abs(y)**2
    kfreq = fftfreq(N)*N

    kfreq2D = np.meshgrid(kfreq, kfreq)
    K = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    K = K.flatten()
    P = P.flatten()
    kbins = np.arange(0.5, N//2, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # Pk, a, b = binned_statistic(K, P, statistic="mean", bins=kbins)
    Pk, bin_edges, _ = binned_statistic(K, P, statistic = "sum", bins = kbins)
    count, _, _ = binned_statistic(K, P, statistic = "count", bins = kbins)
    for i in range(len(Pk)):
        avg_bin = 0.5 * (bin_edges[i] + bin_edges[i+1])
        Pk[i] /= 2*np.pi*avg_bin
    # Pk *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals*2*np.pi/L, Pk, count

def power_spectrum3D(f,N,L):
    y = fftn(f)
    P = np.abs(y)**2
    kfreq = fftfreq(N)*N

    kfreq3D = np.meshgrid(kfreq, kfreq, kfreq)
    K = np.sqrt(kfreq3D[0]**2 + kfreq3D[1]**2 + kfreq3D[2]**2)
    K = K.flatten()
    P = P.flatten()
    kbins = np.arange(0.5, N//2, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # Pk, a, b = binned_statistic(K, P, statistic = "mean", bins = kbins)
    Pk, bin_edges, _ = binned_statistic(K, P, statistic = "sum", bins = kbins)
    count, _, _ = binned_statistic(K, P, statistic = "count", bins = kbins)
    for i in range(len(Pk)):
        avg_bin = 0.5 * (bin_edges[i] + bin_edges[i+1])
        Pk[i] /= 4*np.pi*avg_bin**2
    # Pk *= 4/3 * np.pi * (kbins[1:]**3 - kbins[:-1]**3)

    return kvals*(2*np.pi/L), Pk, count


# helper functions for plotting:

def plot_settings(figsize=(7,5)):
    ''' For nicer plots.
    '''
    font = {'size'   : 20, 'family':'STIXGeneral'}
    axislabelfontsize='large'
    matplotlib.rc('font', **font)
    matplotlib.mathtext.rcParams['legend.fontsize']='small'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.figsize'] = figsize

def plot_field_slice(data, count_id, tau, make_movie=False):
    
    plt.imshow(data, cmap=matplotlib.cm.viridis, interpolation='none')
    plt.xlabel(r"Comoving distance $x/f_a$")
    plt.title(fr" $\tau/\tau_0 = {round(tau, 2)}$", loc = 'left')
    plt.colorbar(fraction=0.05, pad=0.04, shrink=0.75)

    if make_movie:
        plt.savefig("tmp_img_%02d.png" % count_id, facecolor='white')
        plt.clf()
    else:
        plt.show()
    
def plot_axion_slice(data, count_id, tau, make_movie=False):

    plt.imshow(data, cmap=matplotlib.cm.twilight, interpolation='none')
    plt.xlabel(r"Comoving distance $x/f_a$")
    plt.title(fr" $\tau/\tau_0 = {round(tau, 2)}$", loc = 'left')
    cbar = plt.colorbar(fraction=0.05, pad=0.04, shrink=0.75)
    cbar.set_ticks([np.pi-0.001, np.pi/2, 0, -np.pi/2, -np.pi+0.001])
    cbar.set_ticklabels([r"$\pi$", r"$\pi/2$","0",r"$-\pi/2$", r"$-\pi$"])
    cbar.set_label(r"$a(x)/f_a$", rotation=0, labelpad=15)
    
    if make_movie:
        plt.savefig("tmp_img_%02d.png" % count_id, facecolor='white')
        plt.clf()
    else:
        plt.show()

def plot_power_spectrum(data, count_id, tau, make_movie=False):
    ks, pks, count = data

    plt.loglog(ks, pks, "-")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$P\,(k)$", rotation=0, labelpad=30)
    
    if make_movie:
        plt.savefig("tmp_img_%02d.png" % count_id, facecolor='white')
        plt.clf()
    else:
        plt.show()

def convert_format2D(arr, N):
    ret = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ret[i,j] = arr[i + N * j]
    
    return ret

def read_snapshots2D(N, func=lambda p1, p2: np.arctan2(p1, p2)):
    ''' Note: Appends axion field to snapshots by default.
    '''
    snapshots = []
    count = 0
    done = False
    while not done:
        try:
            phi1 = convert_format2D(np.fromfile(f"snapshot{count}-phi1", dtype=np.float64), N)
            phi2 = convert_format2D(np.fromfile(f"snapshot{count}-phi2", dtype=np.float64), N)
            snapshots.append(func(phi1, phi2))
            count += 1
        except:
            done = True
    
    return snapshots

def plot_snapshots2D(N, func=lambda p1, p2: np.arctan2(p1, p2), plotfunc=plot_axion_slice, **plotfunc_kwargs):
    ''' Returns the plot count.
        Note: Plots axion field by default.
    '''
    timings = pd.read_csv("snapshot-timings.csv")
    tau_vals = np.array(timings['tau'].tolist())

    count = 0
    done = False
    while not done:
        try:
            phi1 = convert_format2D(np.fromfile(f"snapshot{count}-phi1", dtype=np.float64), N)
            phi2 = convert_format2D(np.fromfile(f"snapshot{count}-phi2", dtype=np.float64), N)
            plotfunc(func(phi1, phi2), count, tau_vals[count], **plotfunc_kwargs)
            count += 1
        except:
            done = True

    return count

def convert_format3D(arr, N):
    ret = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ret[i,j,k] = arr[(i * N + j) * N + k]

    return ret

def read_snapshots3D(N, func=lambda p1, p2: np.arctan2(p1, p2)):
    ''' Note: Appends axion field to snapshots by default.
    '''
    snapshots = []
    count = 0
    done = False
    while not done:
        try:
            phi1 = convert_format3D(np.fromfile(f"snapshot{count}-phi1", dtype=np.float64), N)
            phi2 = convert_format3D(np.fromfile(f"snapshot{count}-phi2", dtype=np.float64), N)
            snapshots.append(func(phi1, phi2))
            count += 1
        except:
            done = True
    
    return snapshots

def plot_snapshots3D(N, slice_index, func=lambda p1, p2: np.arctan2(p1, p2), plotfunc=plot_axion_slice, **plotfunc_kwargs):
    ''' Returns the plot count.
        Note: Plots axion field by default.
    '''
    timings = pd.read_csv("snapshot-timings.csv")
    tau_vals = np.array(timings['tau'].tolist())

    count = 0
    done = False
    while not done:
        try:
            phi1 = convert_format3D(np.fromfile(f"snapshot{count}-phi1", dtype=np.float64), N)
            phi2 = convert_format3D(np.fromfile(f"snapshot{count}-phi2", dtype=np.float64), N)
            plotfunc(func(phi1, phi2)[slice_index], count, tau_vals[count], **plotfunc_kwargs)
            count += 1
        except:
            done = True

    return count

def run_ffmpeg(filename, framerate=5):

    subprocess.call(['ffmpeg', '-framerate', str(framerate), '-i', 'tmp_img_%02d.png', '-pix_fmt', 'yuv420p', filename])

    # cleanup image files:
    for fname in glob.glob("tmp_img*"):
        os.remove(fname)

if __name__ == '__main__':

    plot_settings(figsize=(13,12))
    count = plot_snapshots2D(256, make_movie=True)
    run_ffmpeg("video.mp4", framerate=count//2)

#     plot_settings(figsize=(13,12))
#     count = plot_snapshots3D(128,128//2, make_movie=True)
#     run_ffmpeg("video.mp4", framerate=count//2)

#     plot_settings((8, 8))
#     plot_snapshots2D(256, func=lambda p1, p2: power_spectrum2D(p1, 256, 256), plotfunc=plot_power_spectrum, make_movie=True)
#     run_ffmpeg("video.mp4", framerate=count//2)

#     plot_settings((8, 8))
#     plot_snapshots3D(128, func=lambda p1, p2: power_spectrum3D(p1, 128, 128), plotfunc=plot_power_spectrum)
#     run_ffmpeg("video.mp4", framerate=count//2)

