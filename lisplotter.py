import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.path as mpath
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import seaborn as sns
import pandas as pd
import numpy as np

class LISPlotter:
    """
    A plotting utility class for LIS model outputs and analysis.
    Provides methods for 2D maps and multi-boxplots.
    Can automatically use lat/lon from cube object if available.
    """

    def __init__(self, cube=None, figsize_map=(8,6), figsize_box=(8,6), dpi=100):
        """
        Parameters
        ----------
        cube : optional
            Cube object that contains `lons` and `lats` attributes.
        figsize_map : tuple
            Default figure size for 2D maps.
        figsize_box : tuple
            Default figure size for boxplots.
        dpi : int
            Figure DPI.
        """
        self.cube = cube
        self.figsize_map = figsize_map
        self.figsize_box = figsize_box
        self.dpi = dpi

        # Attempt to automatically get lons/lats from cube
        self.lons = getattr(cube, 'lons', None)
        self.lats = getattr(cube, 'lats', None)

    def plot_2d_map(self, data, lon=None, lat=None, cmap='viridis', vmin=None, vmax=None, 
                projection='lambert', title='Map', cbar_label='', 
                extent=[-121.07,-70,24.53,50], save_path=None, 
                cent_lon=-97.078, cent_lat=32.5, curve_boundaries=False,
                ax=None, show=True, box_text=None, force_square=False,
                cbar_opts=None,
                land_color='lightgray', ocean_color='#2b8cc9', lake_color='#2b8cc9',
                state_color='black', coast_color='black', border_color='black',
                line_width=0.5, show_borders=True, add_colorbar = True):
        """
        Plot a 2D geospatial map using Cartopy with customizable features, colorbar, and optional subplot handling.

        Parameters
        ----------
        data : xarray.DataArray or np.ndarray
            2D array containing the variable to plot (lat x lon).
        
        lon : np.ndarray, optional
            1D or 2D array of longitudes. If None, will use `self.lons` from the class.
        
        lat : np.ndarray, optional
            1D or 2D array of latitudes. If None, will use `self.lats` from the class.
        
        cmap : str or matplotlib colormap, default 'viridis'
            Colormap for the data.
            Example: cmap='BlAqGrYeOrRe' or cmap=plt.cm.viridis
        
        vmin, vmax : float, optional
            Min and max values for colormap scaling.
            Example: vmin=0, vmax=100
        
        projection : str, default 'lambert'
            Map projection. Options:
            - 'lambert': Lambert Conformal
            - 'platecarree': Plate Carree
        
        title : str, default 'Map'
            Title of the map.
        
        cbar_label : str, default ''
            Label for the colorbar.
        
        extent : list, default [-121.07,-70,24.53,50]
            Geographic extent in form [lon_min, lon_max, lat_min, lat_max].
        
        save_path : str, optional
            File path to save the figure. If None, figure is not saved.
        
        cent_lon, cent_lat : float
            Central longitude and latitude for Lambert projection.
        
        curve_boundaries : bool, default False
            If True, draws a curved boundary using matplotlib Path instead of rectangular extent.
        
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure and axis are created.
        
        show : bool, default True
            If True, calls plt.show() to display the figure immediately.
        
        box_text : str, optional
            Adds an anchored text box to the map (e.g., "(a)", "(b)" for paper figures).
            Example: box_text="(a)"
        
        force_square : bool, default False
            If True, forces the plot to have a square aspect ratio.
        
        cbar_opts : dict, optional
            Additional colorbar options. Keys:
            - 'orientation': 'vertical' or 'horizontal'
            - 'extend': 'neither', 'both', 'min', 'max'
            - 'label': str, overrides `cbar_label`
            - 'width': str or float, width of colorbar (e.g., '2%')
            - 'pad': float, space between plot and colorbar (e.g., 0.03)
        
        land_color, ocean_color, lake_color : str
            Colors for land, ocean, and lakes. Example: land_color='tan', ocean_color='lightblue'
        
        state_color, coast_color, border_color : str
            Colors for state boundaries, coastline, and country borders.
        
        line_width : float, default 0.5
            Line width for boundaries, coastlines, and gridlines.
        
        show_borders : bool, default True
            If True, shows state and country borders.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        
        ax : matplotlib.axes.Axes
            Axis object used for plotting.
        
        pcm : matplotlib.collections.QuadMesh
            The plotted mesh, useful for shared colorbars or further manipulation.

        Examples
        --------
        # Simple plot
        fig, ax, pcm = lisplot.plot_2d_map(data, cmap='viridis', title='Soil Moisture')

        # With specific extent, color limits, and saving
        fig, ax, pcm = lisplot.plot_2d_map(
            data, vmin=0, vmax=100, extent=[-120,-70,25,50], save_path='map.png'
        )

        # Multiple plots with shared colorbar
        fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.LambertConformal()}, figsize=(15,5))
        for i, dat in enumerate([data1, data2, data3]):
            lisplot.plot_2d_map(dat, ax=axs[i], show=False, cbar_opts={'orientation':'vertical','width':'2%'})
        plt.show()

        # Add a box label and force square
        fig, ax, pcm = lisplot.plot_2d_map(
            data, box_text="(a)", force_square=True, cbar_label='mm/day'
        )
        """
        if lon is None:
            if self.lons is None:
                raise ValueError("Longitude array not provided and cube does not have lons.")
            lon = self.lons
        if lat is None:
            if self.lats is None:
                raise ValueError("Latitude array not provided and cube does not have lats.")
            lat = self.lats

        proj = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat) if projection.lower()=='lambert' else ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_map, subplot_kw={'projection': proj}, dpi=self.dpi)
        else:
            fig = ax.figure

        if curve_boundaries:
            rect = mpath.Path([
                [extent[0], extent[2]],
                [extent[1], extent[2]],
                [extent[1], extent[3]],
                [extent[0], extent[3]],
                [extent[0], extent[2]],
            ]).interpolated(20)
            proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
            rect_in_target = proj_to_data.transform_path(rect)
            ax.set_boundary(rect_in_target)
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Plot data
        pcm = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

        # Features with colors
        ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0, alpha=0.8)
        ax.add_feature(cfeature.LAKES, facecolor=lake_color, zorder=0)
        if show_borders:
            ax.add_feature(cfeature.STATES, edgecolor=state_color, alpha=0.5, lw=line_width)
            ax.add_feature(cfeature.COASTLINE, edgecolor=coast_color, lw=line_width)
            ax.add_feature(cfeature.BORDERS, edgecolor=border_color, lw=line_width)

        # Gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False,
                        linewidth=0.33, color='k', alpha=0.5)
        gl.rotate_labels = False
        gl.right_labels = gl.top_labels = False
        gl.xformatter = LongitudeFormatter(number_format=".1f", degree_symbol='°')
        gl.yformatter = LatitudeFormatter(number_format=".1f", degree_symbol='°')
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

        # Colorbar
        cbar_opts = {} if cbar_opts is None else cbar_opts
        orientation = cbar_opts.get("orientation", "vertical")
        extend = cbar_opts.get("extend", "neither")
        label = cbar_opts.get("label", cbar_label)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right" if orientation=="vertical" else "bottom", size=0.2, pad=0.15, axes_class=plt.Axes)
        cbar = plt.colorbar(pcm, cax=cax, orientation=orientation, extend=extend)
        if label is not None:
            if orientation=="vertical":
                cbar.set_label(label, fontsize=12)
            else:
                cbar.set_label(label, fontsize=12)

        # Optional anchored text box
        if box_text is not None:
            anchored_box = AnchoredText(box_text, prop=dict(size=10), frameon=True, loc=2)
            anchored_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(anchored_box)

        # Force square
        if force_square:
            try:
                im = ax.get_images()[0]
                extent_im = im.get_extent()
                ax.set_aspect(abs((extent_im[1]-extent_im[0])/(extent_im[3]-extent_im[2])))
            except IndexError:
                print("Can't force-square an empty axis")

        ax.set_title(title, fontsize=12)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        if show:
            plt.show()

        return fig, ax, pcm

    def plot_multi_boxplots(self, datasets, labels, title="Boxplot", ylabel="Value", xlabel="Experiment",
                            palette=None, figsize=None, show_means=True, flier_style=None, tight_layout=False):
        """
        Plot multiple boxplots using seaborn.
        """
        if figsize is None:
            figsize = self.figsize_box

        flat_data = [np.ravel(ds) for ds in datasets]
        df = pd.DataFrame({label: data for label, data in zip(labels, flat_data)})
        df_melt = df.melt(var_name="Experiment", value_name=ylabel)

        plt.figure(figsize=figsize)

        if palette is None:
            palette_colors = sns.color_palette("Set2", len(labels))
            palette = {label: palette_colors[i] for i, label in enumerate(labels)}

        if flier_style is None:
            flier_style = dict(marker='o', markersize=5, markerfacecolor="orange",
                               markeredgecolor="red", alpha=0.5)

        ax = sns.boxplot(
            x="Experiment",
            y=ylabel,
            data=df_melt,
            width=0.5,
            palette=palette,
            boxprops=dict(edgecolor="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1),
            flierprops=flier_style
        )

        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

        # Annotate means
        if show_means:
            means = df_melt.groupby("Experiment")[ylabel].mean()
            for i, (exp, mean_val) in enumerate(means.items()):
                ax.text(i, mean_val, f"{mean_val:.2f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="black")

        # Legend
        handles = [plt.Rectangle((0,0),1,1, color=palette[exp]) for exp in palette]
        ax.legend(handles, palette.keys(), title="Experiments", loc="best")

        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        if tight_layout:
            plt.tight_layout()

        plt.show()
        return plt.gcf(), ax
