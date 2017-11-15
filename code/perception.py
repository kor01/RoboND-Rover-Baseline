import numpy as np
import cv2
from rover_state import RoverState
from perspective import CalibratedPerception
from perspective import CV2Perception
from geometry import to_polar_coords
from geometry import degree_to_rad
import rover_param as spec


def render_particles(particles):
  particles = particles.transpose()
  particles = particles * spec.DST_SIZE * 2
  particles[:, 1] *= -1
  y_size, x_size = spec.FRAME_SHAPE
  particles[:, 1] += x_size / 2
  particles[:, 0] = y_size - particles[:, 0]
  particles = np.around(particles).astype('uint32')
  particles[:, 1] = particles[:, 1].clip(0, x_size - 1)
  particles[:, 0] = particles[:, 0].clip(0, y_size - 1)
  render = np.zeros(spec.FRAME_SHAPE, dtype=np.uint8)
  render[particles[:, 0], particles[:, 1]] = 255
  return render

def vote_on_mesh(
    values, bins, min_val=None, max_val=None):

  min_val = min_val or values.min()
  max_val = max_val or values.max()
  interval = (max_val - min_val) / bins
  mesh = np.linspace(min_val, max_val, bins + 1)
  predicates = np.abs(mesh[:, None] - values) < interval
  predicates = predicates.astype('int32')
  num_votes = predicates.sum(axis=-1)
  return predicates, num_votes


def densely_distributed(dists, bins):
  _, votes = vote_on_mesh(dists, bins, min_val=1.0)
  return votes.min() > 0


def cluster_to_mesh(values, bins):

  predicates, num_votes = vote_on_mesh(values, bins)

  clusters = []
  current_cluster = set()
  for i in range(len(bins)):
    ids = np.where(predicates[i])[0]
    if len(ids) == 0:
      if len(current_cluster) != 0:
        clusters.append(list(current_cluster))
        current_cluster = set()
    else:
      current_cluster += ids

  if len(current_cluster) > 0:
    clusters.append(list(current_cluster))

  return clusters


def particles_to_ray(particles):
  dists, angles = to_polar_coords(
    particles[0], particles[1])

  # range in (-60, 60)
  threshold = np.pi / 3
  idx = (-threshold < angles < threshold)
  dists, angles = dists[idx], angles[idx]

  # 240 ray proposals, 0.5 degree interval
  proposals = np.linspace(-threshold, threshold, 240 + 1)

  angular_delta = np.abs(proposals[:, None] - angles[None, :])

  votes = angular_delta < (np.pi / 360)

  rays = []
  for i in range(240):
    vote_dist = dists[votes[i, :]]
    if densely_distributed(vote_dist, bins=20):
      rays.append(proposals[i], vote_dist.max())

  return rays


def generate_directions(clusters, rays):
  ret = []
  for cluster in clusters:
    direction = np.mean([rays[x][0] for x in cluster])
    ret.append(direction)
  return ret


CALIBRATED_PERCEPTION = CalibratedPerception()
CV2_PERCEPTION = CV2Perception()


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(rover: RoverState):
  # Perform perception steps to update Rover()
  # TODO:
  # NOTE: camera image is coming to you in Rover.img
  # 1) Define source and destination points for perspective transform
  # 2) Apply perspective transform
  # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
  # 4) Update Rover.vision_image (this will be displayed on left side of screen)
      # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
      #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
      #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

  # 5) Convert map image pixel values to rover-centric coords
  # 6) Convert rover-centric pixel values to world coordinates
  # 7) Update Rover worldmap (to be displayed on right side of screen)
      # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
      #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
      #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

  # 8) Convert rover-centric pixel positions to polar coordinates
  # Update Rover pixel distances and angles
      # Rover.nav_dists = rover_centric_pixel_distances
      # Rover.nav_angles = rover_centric_angles

  w_coords, b_coords = CALIBRATED_PERCEPTION.evaluate(rover)

  front_view = render_particles(b_coords)
  rover.vision_image[:, :, 2] = front_view

  dists, angles = to_polar_coords(b_coords[0], b_coords[1])

  rover.worldmap[w_coords[1], w_coords[0], 2] += 255
  rays = particles_to_ray(b_coords)

  high_rays = [x for x in rays if x[1] > 10]
  clusters = cluster_to_mesh(
    np.array(x[0] for x in high_rays), degree_to_rad(3))

  directions = generate_directions(clusters, high_rays)
  rover.directions = directions

  idx = dists < 20
  dist, angles = dists[idx], angles[idx]

  rover.nav_angles = angles
  rover.nav_dists = dist

  return rover
