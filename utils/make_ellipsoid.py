import numpy as np


def ellipsoid_point(radii, trans, rot_matrix, eps=0.03):
    y, z, x = np.meshgrid(np.arange(-50, 50), np.arange(-50, 50), np.arange(-50, 50))

    ellips = (z / radii[0]) ** 2 + (y / radii[1]) ** 2 + (x / radii[2]) ** 2 < 1 + eps
    ellips = ellips.astype(np.uint8)

    cloud = np.array(np.where(ellips == 1)).T - np.array([50, 50, 50])
    cloud_r = np.dot(cloud, rot_matrix)
    cloud_trans = cloud_r + trans
    out = np.round(cloud_trans)

    return out.astype(np.int16)


def ellipsoid_point_dense(radii, trans, rot_matrix, shape, eps=0.03, margin=1):
    radii = np.asarray(radii, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    rot_matrix = np.asarray(rot_matrix, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.int64)

    if radii.shape != (3,) or trans.shape != (3,) or rot_matrix.shape != (3, 3):
        raise ValueError(
            "radii, trans, and rot_matrix must have shapes (3,), (3,), and (3, 3)"
        )
    if shape.shape != (3,):
        raise ValueError("shape must have shape (3,)")

    radii_safe = np.maximum(np.abs(radii), 1e-6)
    eps_scale = np.sqrt(max(1.0 + eps, 0.0))
    bbox_radii = radii_safe * eps_scale
    extents = np.sqrt(np.sum((bbox_radii[:, None] * rot_matrix) ** 2, axis=0))

    start = np.floor(trans - extents - margin).astype(np.int64)
    stop = np.ceil(trans + extents + margin).astype(np.int64) + 1
    start = np.maximum(start, 0)
    stop = np.minimum(stop, shape)
    if np.any(start >= stop):
        return np.empty((0, 3), dtype=np.int64)

    local_shape = tuple((stop - start).astype(np.int64))
    coords = np.indices(local_shape, dtype=np.float64).reshape(3, -1).T
    coords += start

    local = (coords - trans) @ np.linalg.inv(rot_matrix)
    inside = np.sum((local / radii_safe) ** 2, axis=1) < 1.0 + eps
    return coords[inside].astype(np.int64)
