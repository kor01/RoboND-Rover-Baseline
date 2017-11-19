import abc
import cv2
import numpy as np

from geometry import clip_to_integer
from geometry import color_thresh
from geometry import rotation_matrix_2d
from geometry import translation
from geometry import flip_image_origin
from geometry import degree_to_rad
from geometry import drop_range
from geometry import drop_linear
from geometry import rover_coords
from geometry import pix_to_world

import rover_param as spec
from rover_state import RoverState


def perspective_numerator(e, c, pitch, roll):
  ex, ey, ez = e
  cx, cy, cz = c
  ps, pc = np.sin(pitch), np.cos(pitch)
  rs, rc = np.sin(roll), np.cos(roll)

  w = np.array([[ez * rc * (ez + cx * pc - cz * ps),
                 -ez * (ez + cx * pc - cz * ps) * rs],
                [ez * (pc * (-ey + cy * rc) + (cz - ez * ps) * rs),
                 ez * (rc * (cz - ez * ps) + pc * (ex - cy * rs))]])

  b = np.array([[ez * (pc * (cz * ez - cx * ex * rc + cx * ey * rs) +
                       ps * (cx * ez + cz * ex * rc - cz * ey * rs))],
                [ez * (-(cz * ey + cy * ex * pc) * rc +
                       cy * ez * ps + (-cz * ex + cy * ey * pc) * rs)]])

  return w, b


def perspective_denominator(pitch, roll, e):

  ps, pc = np.sin(pitch), np.cos(pitch)
  rs, rc = np.sin(roll), np.cos(roll)
  ex, ey, ez = e

  w = np.array([ez * pc * rc,  -ez * pc * rs])
  b = np.array([ez * (ez * ps + pc * (-ex * rc + ey * rs))])
  return w, b


def generate_rotation(roll, pitch):
  theta = np.array([pitch, roll])
  (ps, rs), (pc, rc) = np.sin(theta), np.cos(theta)
  rr = np.array([[1, 0, 0], [0, rc, -rs], [0, rs, rc]],
                dtype=np.float64)
  rp = [[pc, 0, ps], [0, 1, 0], [-ps, 0, pc]]
  return np.matmul(rp, rr)


def singular_line(ty, e, pitch, roll):
  ex, ey, ez = e
  ps, pc = np.sin(pitch), np.cos(pitch)
  rs, rc = np.sin(roll), np.cos(roll)
  return ex - ez * ps / pc - (ey - ty) * rs / (rc * pc)


def clip_fit_map(particles):
  return clip_to_integer(particles, spec.WORLD_SIZE)


def pixel_scale_to_frame_scale(quant):
  return quant / float(spec.PIXEL_SCALING)


def create_zyx_perspective():
  horizon_length = float(spec.FRAME_SHAPE[1]) / spec.PIXEL_SCALING
  return ZYXRotatedPerspective(
    spec.CAMERA_POSITION, spec.VIEW_POINT_POSITION, horizon_length)


def unique_particles(particles):
  mp = np.zeros((spec.WORLD_SIZE, spec.WORLD_SIZE), dtype=np.uint8)
  mp[particles[0], particles[1]] = 1
  ret = np.array(mp.nonzero())
  return ret


def to_world_coords(coords, state: RoverState):
  rotation = rotation_matrix_2d(state.yaw)
  w_coords = rotation @ coords
  w_coords = translation(*state.pos, w_coords)
  w_coords = clip_fit_map(w_coords)
  w_coords = unique_particles(w_coords)
  return w_coords


class PerspectiveTransform(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def get_singular(self, roll, pitch):
    pass

  @abc.abstractmethod
  def particle_transform(self, roll, pitch, particles):
    pass


class ZYXRotatedPerspective(PerspectiveTransform):
  def __init__(self, camera_pos, view_pos, horizon_length):
    self._c = camera_pos
    self._e = view_pos
    self._h = horizon_length

  def denominator(self, roll, pitch):
    r = generate_rotation(roll, pitch)
    e = r @ self._e
    w, b = perspective_denominator(pitch, roll, e)
    return w[0], w[1], b[0]

  def get_singular(self, pitch, roll):
    max_x = max([singular_line(0, self._e, pitch, roll),
                 singular_line(self._h, self._e, pitch, roll)])
    return max_x

  def particle_transform(
      self, roll, pitch, particles) -> np.ndarray:

    r = generate_rotation(roll, pitch)
    c, e = r @ self._c, self._e

    w, b = perspective_numerator(e, c, pitch, roll)

    numerator = w @ particles + b

    w, b = perspective_denominator(pitch, roll, e)

    denominator = w @ particles + b
    return numerator / denominator



class CalibratedPerception(object):

  def __init__(self, drop=2):
    self._perspect = create_zyx_perspective()
    self._zyx_singular_drop = drop

  def trans_particles(self, particles, state):

    pitch = degree_to_rad(state.pitch)
    roll = -degree_to_rad(state.roll)
    kx, ky, b = self._perspect.denominator(roll, pitch)
    drop = pixel_scale_to_frame_scale(
      self._zyx_singular_drop)
    particles = drop_linear(particles, kx, ky, b, drop)
    b_coords = self._perspect.particle_transform(
      particles=particles, pitch=pitch, roll=roll)
    b_coords = drop_range(b_coords, axis=0, low=0, high=40)
    b_coords = drop_range(b_coords, axis=1, low=-160, high=160)
    return b_coords


  def evaluate(self, image, state: RoverState):
    coords = flip_image_origin(image)
    particles = np.array(coords.nonzero())
    particles = particles / spec.PIXEL_SCALING
    return self.trans_particles(particles, state)


class CV2Perception(object):

  def __init__(self):
    self._cv2_m = cv2.getPerspectiveTransform(
      spec.STD_PERSPECTIVE_SOURCE, spec.STD_PERSPECTIVE_TARGET)

  def evaluate(self, state: RoverState, segment=None):
    
    if segment is None:
      b_coords = color_thresh(state.img)
      b_coords[:75, :] = 0
    else:
      b_coords = segment

    b_coords = cv2.warpPerspective(
      b_coords, self._cv2_m, (b_coords.shape[1], b_coords.shape[0]))
    b_coords = rover_coords(b_coords)

    scale = spec.DST_SIZE * 2
    b_coords = (b_coords[0] / scale, b_coords[1] / scale)

    # to world coordinate frame
    w_coords = pix_to_world(
      b_coords[0], b_coords[1], state.pos[0],
      state.pos[1], state.yaw, spec.WORLD_SIZE)
    
    return w_coords, b_coords
