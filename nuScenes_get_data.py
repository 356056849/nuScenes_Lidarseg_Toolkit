from nuscenes.nuscenes import NuScenes
import numpy as np
import os
from nuScenes_toolkit import nusc_get_lidar_data
import matplotlib.pyplot as plt
from skimage.io import  imsave


# nuScenes_dataset root directory
data_dir = '/home/ubuntu/MyFiles/wjc/nuScenes/data/sets/nuscenes/'
# initialization
nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
# get one sample
my_sample = nusc.sample[400]
# Print the number of points for each class in the lidar pointcloud of a sample.
nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count')

# extract img
img = nusc_get_lidar_data(nusc, my_sample['token'], 'img')
imsave('img.jpg', img)
# extract vis
ax = nusc_get_lidar_data(nusc, my_sample['token'], 'vis')
plt.savefig('vis.jpg')
# extract depth
depth = nusc_get_lidar_data(nusc, my_sample['token'], 'depth')
np.save('depth.npy', depth)
# extract intensity (0.0 - 1.0)
intensity = nusc_get_lidar_data(nusc, my_sample['token'], 'intensity')
np.save('intensity.npy', intensity)
# extract segLabel
segLabel = nusc_get_lidar_data(nusc, my_sample['token'], 'segLabel')
np.save('segLabel.npy', segLabel)

