"""Execute the shipped notebook's real cells without opening blocking windows."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'tools' / 'scene_segmentation_gcpr16.ipynb'


class NotebookTests(unittest.TestCase):
    def test_notebook_structure_and_code(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        self.assertEqual(notebook['metadata']['kernelspec']['language'], 'python')
        if importlib.util.find_spec('nbformat'):
            import nbformat
            nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{NOTEBOOK.name}:cell{index}', 'exec')
                self.assertIsInstance(cell['outputs'], list)
                self.assertTrue(cell['execution_count'] is None or isinstance(cell['execution_count'], int))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Notebook uses PyTorch tensors')
    def test_sequential_execution_from_tools(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        namespace = {'__name__': '__main__', 'VISUALIZE': False}
        previous_directory, previous_path = Path.cwd(), list(sys.path)
        try:
            os.chdir(ROOT / 'tools')
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), f'{NOTEBOOK.name}:cell{index}', 'exec'), namespace)
            self.assertEqual(namespace['ROOT'], ROOT)
            self.assertEqual(namespace['pts'].shape, (162103, 4))
            self.assertEqual(namespace['gt_box'].shape, (72, 7))
            self.assertGreater(len(namespace['fitted_indices']), 0)
            np.testing.assert_array_equal(namespace['mesh_box_indices'], namespace['fitted_indices'])
            self.assertEqual(len(namespace['rooftop_mask']), len(namespace['mesh_vertices']))
            self.assertTrue(np.isfinite(namespace['mesh_vertices']).all())
            self.assertTrue(namespace['rooftop_mask'].any())
            self.assertGreater(len(namespace['mesh_faces']), 0)
            if importlib.util.find_spec('open3d'):
                # Exercise the actual Open3D conversion and color path without a window.
                import open3d
                import open3d_vis_utils
                geometry = namespace['make_open3d_mesh'](
                    namespace['mesh_vertices'], namespace['mesh_faces'], namespace['vertex_colors'])
                self.assertTrue(geometry.has_vertex_normals())
                np.testing.assert_allclose(np.asarray(geometry.vertex_colors), namespace['vertex_colors'])
                line_set, _ = open3d_vis_utils.translate_boxes_to_open3d_instance(namespace['optimized_boxes'][0])
                self.assertGreater(len(line_set.lines), 0)
        finally:
            os.chdir(previous_directory)
            sys.path[:] = previous_path


if __name__ == '__main__':
    unittest.main()
