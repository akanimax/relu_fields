NUM_COORD_DIMENSIONS = 3  # for (x, y, z)
NUM_COLOUR_CHANNELS = 3  # for (r, g, b)
NUM_RGBA_CHANNELS = 4  # for (r, g, b, a)


SEED = 42  # The answer to life, the universe and everything
ZERO_PLUS = 1e-10
INFINITY = 1e10


# volumetric rendering keys
EXTRA_DISPARITY = "disparity"
EXTRA_ACCUMULATED_WEIGHTS = "accumulated_weight"
EXTRA_POINT_DENSITIES = "point_densities"
EXTRA_POINT_OCCUPANCIES = "point_occupancies"
EXTRA_SAMPLE_INTERVALS = "deltas"
EXTRA_POINT_WEIGHTS = "point_weights"
EXTRA_POINT_DEPTHS = "point_depths"

# camera related keys
CAMERA_BOUNDS = "camera_bounds"
CAMERA_INTRINSICS = "camera_intrinsics"
HEMISPHERICAL_RADIUS = "hemispherical_radius"


# misc keys :D:
EXTRA_INFO = "extra_info"
