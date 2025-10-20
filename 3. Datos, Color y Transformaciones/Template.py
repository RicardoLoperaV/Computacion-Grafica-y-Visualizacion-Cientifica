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
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        
        
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

    def create_academic_plot_type(df, group_col,
                            # --- Core Plotting ---
                            title='Default Title', xlabel='X-axis', ylabel='Y-axis', figsize=(10, 6),
                            plot_type='line',  #'line', 'bar', 'scatter', 'area', 'lollipop'
                            # --- Styling & Colors ---
                            style='whitegrid', palette='viridis', custom_styles=None, linewidth=1.5,
                            # --- Labels & Fonts ---
                            title_fontsize=16, label_fontsize=12, tick_fontsize=10, legend_title=None,
                            # --- Annotations & Layout ---
                            annotations=None, spines_to_remove=['top', 'right'], grid_alpha=0.5,
                            # --- Plot-specific options ---
                            bar_width=0.8, marker_size=6, alpha=0.8):

    
        # 1. Set the overall aesthetic style of the plot
        sns.set_style(style)

        # 2. Create the figure and axes objects with a specified size
        fig, ax = plt.subplots(figsize=figsize)

        # 3. Prepare color and style cycles
        unique_groups = df[group_col].unique()
        colors = sns.color_palette(palette, n_colors=len(unique_groups))

        # Get the numeric columns that represent the x-axis
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        # Convert column names to integers for proper plotting
        x_values = [int(col) for col in numeric_cols if str(col).isdigit()]

        # 4. Plot the data for each group based on plot_type
        if plot_type == 'bar':
            # Calculate bar positions for grouped bars
            num_groups = len(unique_groups)
            bar_width_adjusted = bar_width / num_groups
            
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group]
                y_values = group_data[x_values].mean()
                
                # Offset bars for each group
                x_positions = np.array(x_values) + (i - num_groups/2 + 0.5) * bar_width_adjusted
                
                ax.bar(x_positions, y_values,
                    width=bar_width_adjusted,
                    label=group,
                    color=colors[i],
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=0.5)
        
        elif plot_type == 'lollipop':
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group]
                y_values = group_data[x_values].mean()
                
                # Create lollipop effect: stems + markers
                ax.stem(x_values, y_values, 
                        linefmt=colors[i], 
                        markerfmt='o',
                        basefmt=' ',
                        label=group)
                
                # Customize markers
                lines = ax.get_lines()
                plt.setp(lines[-2:], color=colors[i], linewidth=linewidth, alpha=alpha)
        
        elif plot_type == 'scatter':
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group]
                y_values = group_data[x_values].mean()
                
                linestyle = custom_styles.get(group, '-') if custom_styles else '-'
                
                # Plot line + markers
                ax.plot(x_values, y_values,
                        label=group,
                        color=colors[i],
                        linestyle=linestyle,
                        linewidth=linewidth,
                        marker='o',
                        markersize=marker_size,
                        alpha=alpha)
        
        elif plot_type == 'area':
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group]
                y_values = group_data[x_values].mean()
                
                ax.fill_between(x_values, y_values,
                                label=group,
                                color=colors[i],
                                alpha=alpha * 0.6,
                                linewidth=linewidth,
                                edgecolor=colors[i])
        
        elif plot_type == 'step':
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group]
                y_values = group_data[x_values].mean()
                
                ax.step(x_values, y_values,
                        label=group,
                        color=colors[i],
                        linewidth=linewidth,
                        alpha=alpha,
                        where='mid')
        
        else:  # Default: 'line'
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group]
                y_values = group_data[x_values].mean()
                
                linestyle = custom_styles.get(group, '-') if custom_styles else '-'
                
                ax.plot(x_values, y_values,
                        label=group,
                        color=colors[i],
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha)

        # 5. Customize Labels, Titles, and Ticks with font control
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # 6. Add a Legend with a title
        legend = ax.legend(title=legend_title or group_col, 
                        bbox_to_anchor=(1.04, 1), 
                        loc="upper left",
                        framealpha=0.9)
        plt.setp(legend.get_title(), fontsize=label_fontsize)

        # 7. Add Annotations to highlight key points
        if annotations:
            for ann in annotations:
                ax.annotate(
                    text=ann['text'],
                    xy=ann['xy'],
                    xytext=ann.get('xytext', (20, -20)),
                    textcoords='offset points',
                    arrowprops=ann.get('arrowprops', dict(arrowstyle="->", color='black')),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7)
                )

        # 8. Customize Grids and Spines for a cleaner look
        if style in ['whitegrid', 'darkgrid']:
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=grid_alpha)
        sns.despine(ax=ax, top='top' in spines_to_remove, right='right' in spines_to_remove,
                    left='left' in spines_to_remove, bottom='bottom' in spines_to_remove)

        # 9. Adjust layout to prevent labels from being cut off
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()