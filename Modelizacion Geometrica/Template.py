import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

class Template:
    def __init__(self):
        # Custom style for academic/research plots with vivid colors

        # Configure matplotlib to use LaTeX fonts
        plt.rcParams.update({
        # Font settings - LaTeX style
        'text.usetex': True,  # Use LaTeX for all text rendering
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 23,
        'axes.labelsize': 25,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize':20,
        'figure.titlesize': 23,
        
        # Math text settings - use LaTeX style
        'mathtext.fontset': 'cm',  # Computer Modern font for math
        'mathtext.rm': 'serif',
        'mathtext.it': 'serif:italic',
        'mathtext.bf': 'serif:bold',
        
        
        # Grid settings - darker and more visible
        'grid.alpha': 0.6,
        'grid.color': '#2a2a2a',
        'grid.linewidth': 0.6,
        'grid.linestyle': '-',
        
        # Axes settings
        'axes.grid': True,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.labelweight': 'normal',
        
        # Figure settings
        'figure.facecolor': 'white',
        'axes.facecolor': '#fafafa',
        
        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 6,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': '#333333',
        })

        # Vivid color palette (strong, saturated colors)
        vivid_colors = [
            '#E63946',  # Vivid Red
            '#1D3557',  # Navy Blue
            '#2A9D8F',  # Teal
            '#F77F00',  # Orange
            '#8338EC',  # Purple
            '#06D6A0',  # Mint Green
            '#FFB703',  # Yellow
            '#C1121F',  # Dark Red
            '#0353A4',  # Blue
            '#B5179E',  # Magenta
        ]

        # Set the color cycle for plots
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=vivid_colors)

        # Seaborn integration
        sns.set_palette(vivid_colors)
        sns.set_style("whitegrid", {
            'grid.color': '#2a2a2a',
            'grid.alpha': 0.6,
            'grid.linewidth': 0.6,
        })

    