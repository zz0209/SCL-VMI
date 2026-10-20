import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def resample(image, spacing, interpolator, default):
    size = [int(round(count * step / spacing)) for count, step in zip(image.GetSize(), image.GetSpacing())]
    return sitk.Resample(image, size, sitk.Transform(), interpolator, image.GetOrigin(), (spacing,) * 3, image.GetDirection(), default, image.GetPixelID())


def ball(radius):
    grid = np.indices((2 * radius + 1,) * 3) - radius
    return np.square(grid).sum(0) <= radius * radius


def large_components(mask, minimum, structure=None):
    labels, _ = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.isin(labels, np.flatnonzero(sizes >= minimum))


def lung_region(hu, nodule, settings):
    # Aerated lung, closed and hole-filled so that vessels and the nodule itself belong to the lung.
    radius = settings["closing_mm"]
    lung = np.pad(large_components(hu < settings["lung_hu"], settings["lung_minimum_voxels"]), radius, mode="edge")
    closed = ndimage.binary_closing(lung, structure=ball(radius))[radius:-radius, radius:-radius, radius:-radius]
    return ndimage.binary_fill_holes(closed | nodule)


def surface_contact(surface, region, contact_mm):
    if not region.any():
        border = np.zeros_like(region)
        border[[0, -1]] = border[:, [0, -1]] = border[:, :, [0, -1]] = True
        return float(ndimage.distance_transform_edt(~border)[surface].min()), 0.0
    distance = ndimage.distance_transform_edt(~region)[surface]
    return float(distance.min()), float((distance <= contact_mm).mean())


def contact(hu, nodule, settings):
    surface = nodule & ~ndimage.binary_erosion(nodule, structure=np.ones((3, 3, 3)))
    outside = ~lung_region(hu, nodule, settings) & ~nodule
    structures = large_components((hu > settings["soft_tissue_hu"]) & ~nodule, settings["structure_minimum_voxels"], np.ones((3, 3, 3)))
    boundary_distance, boundary_fraction = surface_contact(surface, outside, settings["contact_mm"])
    structure_distance, structure_fraction = surface_contact(surface, structures, settings["contact_mm"])
    return {"lung_boundary_distance_mm": boundary_distance, "lung_boundary_contact_fraction": boundary_fraction,
            "soft_structure_distance_mm": structure_distance, "soft_structure_contact_fraction": structure_fraction}, outside
