import numpy as np
import rover_param
from geometry import polar_to_cartesian
from geometry import degree_to_rad
from mesh_cluster import cluster_on_mesh


def in_range(values, min_val, max_val):
  ret = np.logical_and(values > min_val, values < max_val)
  return ret


def segment_rock_sample(img):
  r_predicate = in_range(img[:, :, 0], 130, 150)
  g_predicate = in_range(img[:, :, 1], 95, 110)
  b_predicate = in_range(img[:, :, 2], 0, 25)
  ret = np.logical_and(r_predicate, g_predicate)
  ret = np.logical_and(ret, b_predicate)
  ret = ret.astype('uint8')
  return ret


def segment_navigable(img):

  thresh = np.array(rover_param.STD_THRESH)
  shape = [1] * img.ndim
  shape[-1] = -1
  thresh.reshape(shape)
  binary_image = img > thresh
  binary_image = np.all(binary_image, axis=-1)
  binary_image = binary_image.astype('uint8')
  return binary_image


WINDOW_SIZE = 5

def drastic_change_radius(rays, idx):

  if idx < WINDOW_SIZE:
    return False

  if len(rays) - idx < WINDOW_SIZE:
    return False

  left_rays = rays[idx - WINDOW_SIZE: idx]
  right_rays = rays[idx + 1: idx + WINDOW_SIZE + 1]

  max_left = max([x[1][:, 1].max() for x in left_rays])
  max_right = max(x[1][:, 1].max() for x in right_rays)

  this_ray_length = rays[idx][1][0, 1]

  if max_left / this_ray_length > 1.5 \
    and max_right / this_ray_length > 1.5:
    return True


def detect_obstacles(rays):
  ret = []
  # segmented ray, drastic radius change are features for obstacles
  for i, ray in enumerate(rays):
    if len(ray.segments) > 0:
      ret.append(polar_to_cartesian(ray.segments[0, 1], ray.theta))
    elif drastic_change_radius(rays, i):
      ret.append(polar_to_cartesian(ray.segments[0, 1], ray.theta))
  return ret


DIRECTION_DROP_RADIUS = 10
ANGLE_INTERVAL = degree_to_rad(1)

def detect_directions(rays):
  # drop segmented rays
  rays = [r for r in rays if len(r.segment) == 1]

  # drop short term rays
  rays = [r for r in rays if r[0, 1] < DIRECTION_DROP_RADIUS]

  # mesh cluster on interval = 1 degree
  angles = np.array([r.theta for r in rays])
  clusters = cluster_on_mesh(angles, interval=ANGLE_INTERVAL)

  ret = []
  for cluster in clusters:
    ret.append(angles[cluster].mean())

  return ret
