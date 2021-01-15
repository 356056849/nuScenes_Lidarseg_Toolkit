from typing import List, Tuple, Dict, Iterable
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from nuscenes.lidarseg.lidarseg_utils import get_labels_in_coloring, create_lidarseg_legend
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
import numpy as np
import cv2
import os
import os.path as osp


def colormap_to_colors(colormap: Dict[str, Iterable[int]], name2idx: Dict[str, int]) -> np.ndarray:
    """
    Create an array of RGB values from a colormap. Note that the RGB values are normalized
    between 0 and 1, not 0 and 255.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :param name2idx: A dictionary containing the mapping form class names to class index.
    :return: An array of colors.
    """
    colors = []
    for i, (k, v) in enumerate(colormap.items()):
        # Ensure that the indices from the colormap is same as the class indices.
        assert i == name2idx[k], 'Error: {} is of index {}, ' \
                                 'but it is of index {} in the colormap.'.format(k, name2idx[k], i)
        colors.append(v)
    # -------------------------------------->
    # -------------------------------------->
    # -------------------------------------->
    # colors = np.array(colors) / 255  # Normalize RGB values to be between 0 and 1 for each channel.
    colors = np.array(colors) / 255
    # -------------------------------------->
    # -------------------------------------->
    # -------------------------------------->
    return colors


def filter_colors(colors: np.array, classes_to_display: np.array) -> np.ndarray:
    """
    Given an array of RGB colors and a list of classes to display, return a colormap (in RGBA) with the opacity
    of the labels to be display set to 1.0 and those to be hidden set to 0.0
    :param colors: [n x 3] array where each row consist of the RGB values for the corresponding class index
    :param classes_to_display: An array of classes to display (e.g. [1, 8, 32]). The array need not be ordered.
    :return: (colormap <np.float: n, 4)>).

    colormap = np.array([[R1, G1, B1],             colormap = np.array([[1.0, 1.0, 1.0, 0.0],
                         [R2, G2, B2],   ------>                        [R2,  G2,  B2,  1.0],
                         ...,                                           ...,
                         Rn, Gn, Bn]])                                  [1.0, 1.0, 1.0, 0.0]])
    """
    for i in range(len(colors)):
        if i not in classes_to_display:
            colors[i] = [1.0, 1.0, 1.0]  # Mask labels to be hidden with 1.0 in all channels.

    # Convert the RGB colormap to an RGBA array, with the alpha channel set to zero whenever the R, G and B channels
    # are all equal to 1.0.
    alpha = np.array([~np.all(colors == 1.0, axis=1) * 1.0])
    colors = np.concatenate((colors, alpha.T), axis=1)

    return colors


def paint_points_label(is_color, lidarseg_labels_filename: str, filter_lidarseg_labels: List[int],
                       name2idx: Dict[str, int], colormap: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    Paint each label in a pointcloud with the corresponding RGB value, and set the opacity of the labels to
    be shown to 1 (the opacity of the rest will be set to 0); e.g.:
        [30, 5, 12, 34, ...] ------> [[R30, G30, B30, 0], [R5, G5, B5, 1], [R34, G34, B34, 1], ...]
    :param lidarseg_labels_filename: Path to the .bin file containing the labels.
    :param filter_lidarseg_labels: The labels for which to set opacity to zero; this is to hide those points
                                   thereby preventing them from being displayed.
    :param name2idx: A dictionary containing the mapping from class names to class indices.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :return: A numpy array which has length equal to the number of points in the pointcloud, and each value is
             a RGBA array.
    """

    # Load labels from .bin file.
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]

    # Given a colormap (class name -> RGB color) and a mapping from class name to class index,
    # get an array of RGB values where each color sits at the index in the array corresponding
    # to the class index.
    colors = colormap_to_colors(colormap, name2idx)  # Shape: [num_class, 3]

    if filter_lidarseg_labels is not None:
        # Ensure that filter_lidarseg_labels is an iterable.
        assert isinstance(filter_lidarseg_labels, (list, np.ndarray)), \
            'Error: filter_lidarseg_labels should be a list of class indices, eg. [9], [10, 21].'

        # Check that class indices in filter_lidarseg_labels are valid.
        assert all([0 <= x < len(name2idx) for x in filter_lidarseg_labels]), \
            'All class indices in filter_lidarseg_labels should ' \
            'be between 0 and {}'.format(len(name2idx) - 1)

        # Filter to get only the colors of the desired classes; this is done by setting the
        # alpha channel of the classes to be viewed to 1, and the rest to 0.
        colors = filter_colors(colors, filter_lidarseg_labels)  # Shape: [num_class, 4]

    # Paint each label with its respective RGBA value.
    coloring = colors[points_label]  # Shape: [num_points, 4]

    return coloring if is_color else points_label


def map_pointcloud_to_image(nusc_explorer,
                            is_color,  # True: return label colors else return label index
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = nusc_explorer.nusc.get('sample_data', camera_token)
    pointsensor = nusc_explorer.nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc_explorer.nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        if show_lidarseg:
            assert hasattr(nusc_explorer.nusc, 'lidarseg'), 'Error: nuScenes-lidarseg not installed!'

            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

            assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
                                         'render_intensity or show_lidarseg, not both.'

        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc_explorer.nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc_explorer.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc_explorer.nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc_explorer.nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc_explorer.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']

        if lidarseg_preds_bin_path:
            sample_token = nusc_explorer.nusc.get('sample_data', pointsensor_token)['sample_token']
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
        else:
            if len(nusc_explorer.nusc.lidarseg) > 0:  # Ensure lidarseg.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(nusc_explorer.nusc.dataroot,
                                                    nusc_explorer.nusc.get('lidarseg', pointsensor_token)['filename'])
            else:
                lidarseg_labels_filename = None

        if lidarseg_labels_filename:
            # Paint each label in the pointcloud with a RGBA value.
            coloring = paint_points_label(is_color, lidarseg_labels_filename, filter_lidarseg_labels,
                                          nusc_explorer.nusc.lidarseg_name2idx_mapping, nusc_explorer.nusc.colormap)
        else:
            coloring = depths
            print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                  'from the ego vehicle instead.'.format(nusc_explorer.nusc.version))
    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im


def nusc_get_lidar_data(nusc, sample_token : str, data_type : str, camera_channel='CAM_FRONT'):
    assert data_type in ['img', 'segLabel', 'depth', 'intensity', 'vis'], '%s is not a valid data type.' % data_type

    sample_record = nusc.explorer.nusc.get('sample', sample_token)
    pointsensor_token = sample_record['data']['LIDAR_TOP']
    camera_token = sample_record['data'][camera_channel]

    if data_type == 'img':
        is_color, is_render_intensity, is_show_lidarseg = False, False, False
    elif data_type == 'vis':
        is_color, is_render_intensity, is_show_lidarseg = True, False, True
    elif data_type == 'depth':
        is_color, is_render_intensity, is_show_lidarseg = False, False, False
    elif data_type == 'intensity':
        is_color, is_render_intensity, is_show_lidarseg = False, True, False
    else:  # data_type == 'segLabel'
        is_color, is_render_intensity, is_show_lidarseg = False, False, True

    points, coloring, im = map_pointcloud_to_image(nusc.explorer, is_color, pointsensor_token, camera_token,
                                                   render_intensity=is_render_intensity,
                                                   show_lidarseg=is_show_lidarseg,
                                                   filter_lidarseg_labels=None,
                                                   lidarseg_preds_bin_path=None)

    dot_size = 5
    im = np.array(im)
    if data_type == 'img':
        return im
    elif data_type == 'vis':
        # Init axes.
        im = np.array(im)
        fig = plt.figure(figsize=(im.shape[1] / 100., im.shape[0] / 100.), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        fig.canvas.set_window_title(sample_token)
        ax.axis('off')
        ax.imshow(im)
        ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)  # c should be within 0-1 range
        color_legend = colormap_to_colors(nusc.explorer.nusc.colormap, nusc.explorer.nusc.lidarseg_name2idx_mapping)
        filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring)
        create_lidarseg_legend(filter_lidarseg_labels,
                               nusc.explorer.nusc.lidarseg_idx2name_mapping, nusc.explorer.nusc.colormap)
        return ax
    elif data_type == 'depth' or data_type == 'intensity':
        background = np.zeros_like(im)[:, :, 0].astype('float32')
        for i in range(len(points[0, :])):
            background[round(points[1, i]), round(points[0, i])] = coloring[i]
        return background

    elif data_type == 'segLabel':
        background = np.zeros_like(im)[:, :, 0].astype('uint8')
        for i in range(len(points[0, :])):
            background[round(points[1, i]), round(points[0, i])] = coloring[i]
        return background

    else:
        return im


