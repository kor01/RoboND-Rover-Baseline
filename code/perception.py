import numpy as np
from matplotlib import pyplot as plt
from rover_state import RoverState
from perspective import CalibratedPerception
from perspective import CV2Perception
from geometry import to_polar_coords
from geometry import polar_to_cartesian
import rover_param as spec
from collections import namedtuple
from ray_detect import particle_to_rays
from perspective import to_world_coords
from object_detect import detect_directions
from object_detect import segment_rock_sample
from object_detect import segment_navigable


def add_to_set(coll, delta):
  for x in delta:
    coll.add(x)


def show_image(image):
  plt.imshow(image)
  plt.show()



RenderRay = namedtuple('RenderRay', ('theta', 'segments'))

def render_directions(directions):
  ret = []
  for direction in directions:
    augmented = np.linspace(direction - 0.02, direction + 0.02, 10)
    segments = np.array([[0.0, 10.0]])
    rays = [RenderRay(theta=x, segments=segments) for x in augmented]
    ret.extend(rays)

  render = render_rays(ret)
  return render


def render_particles(particles):
  render = np.zeros(spec.FRAME_SHAPE, dtype=np.uint8)
  if particles.size == 0:
    return render
  
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



def rays_to_particles(rays, interval):
  particles = []
  for ray in rays:
    for segment in ray.segments:
      bins = int((segment[1] - segment[0]) / interval)
      if bins == 0:
        continue
      quants = np.linspace(segment[0], segment[1], bins)
      coords = polar_to_cartesian(quants, ray.theta)
      particles.extend(coords.transpose())
  particles = np.array(particles)
  return particles.transpose()


def render_rays(rays, interval=0.1):
  particles = rays_to_particles(rays, interval)
  return render_particles(particles)


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


  navi_segment = segment_navigable(rover.img)
  b_navi = CALIBRATED_PERCEPTION.evaluate(navi_segment, rover)
  r, theta = to_polar_coords(b_navi[0], b_navi[1])
  rover.nav_dists = r
  rover.nav_angles = theta
  w_navi = to_world_coords(b_navi, rover)
  #rover.vision_image[:, :, 2] = render_particles(b_navi)
  #rover.worldmap[w_navi[1], w_navi[0], 2] += 1
  
  rays = particle_to_rays(
     r, theta, np.pi / 3, min_dist=0.5, max_dist=9.8)

  b_particles = rays_to_particles(rays, 0.1)
  
  rover.vision_image[:, :, 2] = render_particles(b_particles)

  if b_particles.size > 0:
    w_navi = to_world_coords(b_particles, rover)
    rover.worldmap[w_navi[1], w_navi[0], 2] += 1

  directions = detect_directions(rays)

  rock_sample = segment_rock_sample(rover.img)
  b_samples = CALIBRATED_PERCEPTION.evaluate(rock_sample, rover)
  rover.vision_image[:, :, 1] = render_particles(b_samples)

  w_samples = to_world_coords(b_samples, rover)

  if w_samples.size > 0:
    print('rock_coord:', w_samples)
    rover.worldmap[w_samples[1], w_samples[0], 1] += 1


  rover.directions = directions
  
  #rover.directions = []


  return rover
