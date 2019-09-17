import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from osgeo import gdal, gdal_array

filename = "test1.tif" #path to raster
img_ds = gdal.Open(filename)
array = img_ds.ReadAsArray()
colormap = LinearSegmentedColormap.from_list('mycmap', [(.5, .5, .5), (1, 0.82745, 0), (.14902, .43921, 0)])
plt.figure(figsize=(150,150))
fig1 = plt.imshow(array, cmap = colormap)
plt.axis('off')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.savefig('results.png', bbox_inches='tight', pad_inches=0)