import numpy as np
import cv2
from rover_state import RoverState
from perspective import CalibratedPerception
from perspective import CV2Perception
from geometry import to_polar_coords
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

  idx = dists < 20
  dist, angles = dists[idx], angles[idx]

  rover.nav_angles = angles
  rover.nav_dists = dist

  return rover
