"""Drop-in names for the existing scene_segmentation imports."""
from ..scene import GCPR16Scene, GCPR16Vehicles, point_cloud_scene, vehicle_object

__all__ = ['point_cloud_scene', 'vehicle_object', 'GCPR16Scene', 'GCPR16Vehicles']
