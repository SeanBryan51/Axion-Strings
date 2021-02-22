#================================Plot.py==================================#
# Created by Giovanni Pierobon 2021

# Description:
# Plots for Axion Simulations 

#==============================================================================#

from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

def Snap_Save(fig,index,mov_dir):
    fig.savefig(mov_dir+str(index)+'.png',bbox_inches='tight')

def cbar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
            minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")
    return cbar

def APlot(field,index,lw=2.5,lfs=30,tfs=25):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    fig = plt.figure(figsize=(13,12))
    ax = fig.add_subplot(111)
    im=ax.imshow(field,cmap=cm.twilight,origin='lower')
    ax.set_title(r'$\hat{\tau} = %d$' % index)
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    cb = cbar(im)
    return fig,ax

def SPlot(field,index,lw=2.5,lfs=30,tfs=25):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    fig = plt.figure(figsize=(13,12))
    ax = fig.add_subplot(111)
    im=ax.imshow(field,cmap=cm.binary,vmin=0,vmax=1,origin='lower')
    ax.set_title(r'$\hat{\tau} = %d$' % index)
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    cb = cbar(im)
    return fig,ax

def AxionBar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
            minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")
    cbar.set_label(r'$a/f_a$',rotation=0,labelpad=30,fontsize=30)
    return cbar

def SaxionBar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
            minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")
    cbar.set_label(r'$\vert\Phi\vert/f_a$',rotation=0,labelpad=30,fontsize=30)

    return cbar

def AxionPlot(field,index,xlab=r'$L/L_1$',ylab=r'$L/L_1$',\
                 lw=2.5,lfs=30,tfs=25,size_x=13,size_y=12,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)

    ax.imshow(field,cmap=cm.twilight,origin='lower')
    ax.set_title(r'$\hat{\tau} = %d$' % index)
    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)
    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    if Grid:
        ax.grid()
    return fig,ax

def SaxionPlot(field,index,xlab=r'$L/L_1$',ylab='',\
                 lw=2.5,lfs=30,tfs=25,size_x=13,size_y=12,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)

    ax.imshow(field,cmap=cm.binary,vmin=0,vmax=1,origin='lower')
    ax.set_title(r'$\hat{\tau} = %d$' % index)
    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)
    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    if Grid:
        ax.grid()
    return fig,ax

def DoublePlot(xlab1='L/L_1',ylab1='',xlab2='',ylab2='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=11,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    #mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']
    fig, axarr = plt.subplots(1, 2,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax1.set_title(r'$\hat{\tau} = %d$' % i)
    ax2.set_title(r'$\hat{\tau} = %d$' % i)
    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)
    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
    return fig,ax1,ax2

#def ScalingPlot(filename1,xlab=r'$\hat{\tau}$',ylab=r'$\xi$',lw=2.5,lfs=30,tfs=25,size_x=10,size_y=7,Grid=False):
#    with open(filename1, 'r') as f1:
#        lines1 = f1.readlines()
#        x1 = [float(line.split()[0]) for line in lines1]
#        y1 = [float(line.split()[1]) for line in lines1]
#    plt.rcParams['axes.linewidth'] = lw
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif',size=tfs)
#	fig = plt.figure(figsize=(size_x,size_y))
#    ax = fig.add_subplot(111)
#    ax.plot(x1,y1)
#    plt.xscale('log')
#    ax.set_xlabel(xlab,fontsize=lfs)
#    ax.set_ylabel(ylab,fontsize=lfs,rotation=0,labelpad=20)
#    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
#    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
#    if Grid:
#        ax.grid()
#    return fig,ax
    