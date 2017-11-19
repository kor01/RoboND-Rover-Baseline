import numpy as np
from rover_state import RoverState
from perspective import CalibratedPerception
from perspective import CV2Perception
from geometry import to_polar_coords
from visualization import render_particles
from visualization import rays_to_particles
from ray_detect import particle_to_rays
from perspective import to_world_coords
from object_detect import detect_directions
from object_detect import segment_rock_sample
from object_detect import segment_navigable
from object_detect import segment_obstacles


CALIBRATED_PERCEPTION = CalibratedPerception()
CV2_PERCEPTION = CV2Perception()



def perception_step(rover: RoverState):

  navi_segment = segment_navigable(rover.img)
  b_navi = CALIBRATED_PERCEPTION.evaluate(navi_segment, rover)
  r, theta = to_polar_coords(b_navi[0], b_navi[1])

  rover.nav_dists = r
  rover.nav_angles = theta
  # w_navi = to_world_coords(b_navi, rover)
  # rover.vision_image[:, :, 2] = render_particles(b_navi)
  # rover.worldmap[w_navi[1], w_navi[0], 2] += 1
  
  rays = particle_to_rays(
     r, theta, np.pi / 3, min_dist=0.5, max_dist=9.8)

  b_navi = rays_to_particles(rays, 0.1)
  
  rover.vision_image[:, :, 2] = render_particles(b_navi)

  if b_navi.size > 0:
    w_navi = to_world_coords(b_navi, rover)
    rover.worldmap[w_navi[1], w_navi[0], 2] = 255

  obstacles = segment_obstacles(rover.img)
  b_obstacles = CALIBRATED_PERCEPTION.evaluate(obstacles, rover)
  rover.vision_image[:, :, 0] = render_particles(b_obstacles)
  w_obstacles = to_world_coords(b_obstacles, rover)
  rover.worldmap[w_obstacles[1], w_obstacles[0], 1] = 255

  directions = detect_directions(rays)

  rock_sample = segment_rock_sample(rover.img)
  b_samples = CALIBRATED_PERCEPTION.evaluate(rock_sample, rover)
  rover.vision_image[:, :, 1] = render_particles(b_samples)

  w_samples = to_world_coords(b_samples, rover)

  if w_samples.size > 0:
    rover.worldmap[w_samples[1], w_samples[0], 1] += 1

  rover.directions = directions

  return rover
