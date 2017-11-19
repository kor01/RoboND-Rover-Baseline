import numpy as np
from geometry import polar_to_cartesian
import rover_param as param
from collections import namedtuple
from matplotlib import pyplot as plt


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
  render = np.zeros(param.FRAME_SHAPE, dtype=np.uint8)
  if particles.size == 0:
    return render

  particles = particles.transpose()
  particles = particles * param.DST_SIZE * 2
  particles[:, 1] *= -1
  y_size, x_size = param.FRAME_SHAPE
  particles[:, 1] += x_size / 2
  particles[:, 0] = y_size - particles[:, 0]
  particles = np.around(particles).astype('uint32')
  particles[:, 1] = particles[:, 1].clip(0, x_size - 1)
  particles[:, 0] = particles[:, 0].clip(0, y_size - 1)
  render = np.zeros(param.FRAME_SHAPE, dtype=np.uint8)
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
