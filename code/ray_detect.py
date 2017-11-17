import numpy as np
from mesh_cluster import cluster_on_mesh


VIEW_ANGLE = np.pi / 3
NUM_RAY_MESH = 240


def interpolate_ray(
    angle, dists, focal_length, focal_height, interval):

  # projection ray to 2D plane
  focal_length = focal_length / np.cos(angle)

  proj_ratio = dists / (dists + focal_length)
  proj_dists = focal_height * proj_ratio
  print('project_dim', proj_dists.min(), proj_dists.max())
  # find gap by mesh in projection
  # mesh cluster the projected dists

  clusters = cluster_on_mesh(proj_dists, interval=interval)

  ret = []
  for cluster in clusters:
    start, end = dists[cluster].min(), dists[cluster].max()
    ret.append((start, end))
  print('num_segs:', len(ret))
  return np.array(ret)


FOCAL_LENGTH = 5
FOCAL_HEIGHT = 10
INTERVAL = 0.3

class Ray(object):

  def __init__(self, theta, dists):
    self.theta = theta
    segments = interpolate_ray(
      theta, dists, FOCAL_LENGTH, FOCAL_HEIGHT, INTERVAL)
    self.segments = segments


def particle_to_rays(
    dists, angles, max_angle, min_dist, max_dist):

  angle_idx = np.logical_and(angles > -max_angle, angles < max_angle)
  dist_idx = np.logical_and(dists > min_dist, dists < max_dist)
  idx = np.logical_and(angle_idx, dist_idx)

  dists, angles = dists[idx], angles[idx]

  proposals = np.linspace(
    -max_angle, max_angle, NUM_RAY_MESH + 1)

  angular_delta = np.abs(proposals[:, None] - angles[None, :])

  interval = 2.0 * VIEW_ANGLE / NUM_RAY_MESH
  votes = angular_delta < interval

  ret = []
  for i in range(NUM_RAY_MESH):
    vote_dist = dists[votes[i, :]]
    if len(vote_dist) <= 1:
      continue
    print('vote_dims:', proposals[i],
          vote_dist.min(), vote_dist.max(), len(vote_dist))
    ret.append(Ray(proposals[i], vote_dist))

  return ret
