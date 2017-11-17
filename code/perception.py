import numpy as np
from rover_state import RoverState
from perspective import CalibratedPerception
from perspective import CV2Perception
from geometry import to_polar_coords
from geometry import polar_to_cartesian
import rover_param as spec

def add_to_set(coll, delta):
  for x in delta:
    coll.add(x)


def render_rays(rays):
  particles = []
  for ray in rays:
    


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
  if len(dists) == 0:
    return False
  _, votes = vote_on_mesh(dists, bins, min_val=1.0)
  return votes.min() > 0


def cluster_to_mesh(values, bins):

  if len(values) == 0:
    return []
  
  predicates, num_votes = vote_on_mesh(values, bins)

  clusters = []
  current_cluster = set()
  for i in range(bins):
    ids = np.where(predicates[i])[0]
    if len(ids) == 0:
      if len(current_cluster) != 0:
        clusters.append(list(current_cluster))
        current_cluster = set()
    else:
      add_to_set(current_cluster, ids)

  if len(current_cluster) > 0:
    clusters.append(list(current_cluster))

  return clusters


def particles_to_ray(particles):
  
  dists, angles = to_polar_coords(
    particles[0], particles[1])

  # range in (-60, 60)
  threshold = np.pi / 3
  idx = np.logical_and(-threshold < angles, angles < threshold)
  
  dists, angles = dists[idx], angles[idx]
  
  # 240 ray proposals, 0.5 degree interval
  proposals = np.linspace(-threshold, threshold, 240 + 1)

  angular_delta = np.abs(proposals[:, None] - angles[None, :])

  votes = angular_delta < (np.pi / 360)

  rays = []
  for i in range(240):
    vote_dist = dists[votes[i, :]]
    if len(vote_dist) == 0:
      continue
    if densely_distributed(vote_dist, bins=20):
      rays.append((proposals[i], vote_dist.max()))

  return np.array(rays)


def generate_directions(clusters, rays):
  ret = []
  for cluster in clusters:
    direction = np.mean([rays[x][0] for x in cluster])
    ret.append(direction)
  return ret


def in_range(values, min_val, max_val):
  ret = np.logical_and(values > min_val, values < max_val)
  return ret


def segment_rock_sample(img):
  r_predicate = in_range(img[:, :, 0], 130, 150)
  g_predicate = in_range(img[:, :, 1], 95, 110)
  b_predicate = in_range(img[:, :, 2], 0, 25)
  ret = np.logical_and(r_predicate, g_predicate)
  ret = np.logical_and(ret, b_predicate)
  return ret.astype('uint8')


def extract_rock_pos(segment):
  y, x = segment.nonzero()

  if len(x) == 0:
    return None
  
  idx = y.argmin()
  print(y, x)
  return x[idx], y[idx]


def extract_obstacles(rays, bound):
  # get ride of the rays ends of max_range

  r, theta = rays[:, 1], rays[:, 0]
  num_low = (r > 5).sum()
  ratio = float(num_low) / len(rays)

  # unlikely to be obstacles, more likely to be road edge
  if ratio > 0.8:
    return []

  short_rays = rays[r < bound]
  # get ride of very short ray
  short_rays = short_rays[r > 0.5]
  short_r, short_theta = short_rays[:, 1], short_rays[:, 0]
  edge_points = polar_to_cartesian(short_r, short_theta)
  return edge_points


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

  rover.worldmap[w_coords[1], w_coords[0], 2] += 1
  
  rays = particles_to_ray(b_coords)

  high_rays = rays[rays[:, 1] > 8, :]
  high_angles = high_rays[:, 0]
  clusters = cluster_to_mesh(high_angles, 8)
  
  if len(clusters) > 1:
    directions = generate_directions(clusters, high_rays)
    #print(directions)
    rover.directions = directions
  else:
    rover.directions = None

  rock_sample = segment_rock_sample(rover.img)
  
  rover.vision_image[:, :, 1] = rock_sample
  w_sample, _ = CV2_PERCEPTION.evaluate(
    rover, segment=rock_sample)
  
  if w_sample.size > 0:
    print('rock_coord:', w_sample)
    rover.worldmap[w_sample[1], w_sample[0], 1] += 1

  obstacles = extract_obstacles(rays, 5)
  rover.worldmap[obstacles[1], obstacles[0], 0] += 1

  dists, angles = to_polar_coords(b_coords[0], b_coords[1])
  idx = dists < 20
  dist, angles = dists[idx], angles[idx]

  rover.nav_angles = angles
  rover.nav_dists = dist

  return rover
