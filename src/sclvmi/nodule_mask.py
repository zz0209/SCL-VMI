import numpy as np
from scipy import ndimage


def target_component(mask):
    labels, count = ndimage.label(mask > 0, structure=np.ones((3, 3, 3)))
    assert count > 0, "Mask has no foreground"
    center = tuple(np.asarray(mask.shape) // 2)
    chosen = labels[center]
    if chosen == 0:
        points = np.argwhere(labels > 0)
        chosen = labels[tuple(points[np.argmin(np.square(points - np.asarray(center)).sum(1))])]
    return labels == chosen, count
