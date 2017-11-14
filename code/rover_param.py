import numpy as np


FRAME_SHAPE = (160, 320)
FRAME_ORIGIN = (FRAME_SHAPE[0], FRAME_SHAPE[1]/2)

DST_SIZE = 5
BOTTOM_OFFSET = 6
WORLD_SIZE = 200

# perspective source and target when the roll and pitch = 0

STD_PERSPECTIVE_SOURCE = \
  np.float32([[14.32, 140.71], [120.78, 95.5],
              [199.49, 96.84], [302.7, 140.71]])

STD_PERSPECTIVE_TARGET = \
  np.float32([[FRAME_SHAPE[1]/2 - DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET],
              [FRAME_SHAPE[1]/2 - DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET - 2 * DST_SIZE],
              [FRAME_SHAPE[1]/2 + DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET - 2 * DST_SIZE],
              [FRAME_SHAPE[1]/2 + DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET]])

STD_THRESH = np.uint32([160, 160, 160])

DEFAULT_THRESHOLD = np.float32([160, 160, 160])

# evaluated from calibration optimization
CAMERA_POSITION = np.array(
  [0.27883144,  0.07262362,  0.17211595], dtype=np.float64)
CAMERA_POSITION.flags.writeable = False

# evaluated from calibration optimization
VIEW_POINT_POSITION = np.array(
  [0.04027264,  0.08037242, -0.05411853], dtype=np.float64)
VIEW_POINT_POSITION.flags.writeable = False

# guess value
PIXEL_SCALING = 2000
