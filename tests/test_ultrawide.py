from paths import ROOT
import struct
import unittest
from patches import (ULTRAWIDE_PATCHES, ultrawide_patch, ultrawide_writes,
                     SCENE_WIDTH, SCENE_HEIGHT, UI_WIDTH, UI_HEIGHT)
from test_patches import immediate

class UltrawideTests(unittest.TestCase):
    def test_wide_ui_with_retail_scene_and_heaps(self):
        for size, name in ULTRAWIDE_PATCHES.items():
            with self.subTest(size=size):
                width, height = map(int, size.split('x'))
                self.assertEqual(ultrawide_patch({'ultrawide_res': size}), name)
                writes = ultrawide_writes(ROOT / 'patches/Bloodborne.xml', name,
                                         '01.09', [(0, 0x6000000)])
                for addr, length in [(SCENE_WIDTH, 9), (SCENE_HEIGHT, 9),
                                     (0x04B36DDA - 0x400000, 2), (0x04B36DE2 - 0x400000, 2)]:
                    self.assertFalse(any(a < addr + length and a + len(d) > addr for a, d in writes))
                self.assertEqual(immediate(writes, UI_WIDTH)[1], width)
                self.assertEqual(immediate(writes, UI_HEIGHT)[1], height)
                camera = dict(writes)[0x0183A35D - 0x400000]
                self.assertAlmostEqual(struct.unpack('<f', camera)[0], width / height, places=5)

    def test_off_and_invalid_resolution(self):
        self.assertEqual(ultrawide_patch({}), '')
        self.assertEqual(ultrawide_patch({'ultrawide_res': ''}), '')
        with self.assertRaises(ValueError):
            ultrawide_patch({'ultrawide_res': '3000x1000'})
