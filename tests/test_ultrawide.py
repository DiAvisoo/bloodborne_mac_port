from paths import ROOT
import unittest
from patches import (ULTRAWIDE_PATCHES, ultrawide_patch, compile_patches,
                     SCENE_WIDTH, SCENE_HEIGHT, UI_WIDTH, UI_HEIGHT)
from test_patches import immediate

class UltrawideTests(unittest.TestCase):
    def test_native_scene_and_ui_for_every_selection(self):
        for size, name in ULTRAWIDE_PATCHES.items():
            with self.subTest(size=size):
                width, height = map(int, size.split('x'))
                self.assertEqual(ultrawide_patch({'ultrawide_res': size}), name)
                writes = compile_patches(ROOT / 'patches/Bloodborne.xml', [name], '01.09', [(0, 0x6000000)])
                self.assertEqual(immediate(writes, SCENE_WIDTH)[1], width)
                self.assertEqual(immediate(writes, SCENE_HEIGHT)[1], height)
                self.assertEqual(immediate(writes, UI_WIDTH)[1], width)
                self.assertEqual(immediate(writes, UI_HEIGHT)[1], height)
                # The camera aspect correction must accompany the viewport dimensions.
                self.assertTrue(any(offset == 0x0183A35D - 0x400000 for offset, _ in writes))

    def test_off_and_invalid_resolution(self):
        self.assertEqual(ultrawide_patch({}), '')
        self.assertEqual(ultrawide_patch({'ultrawide_res': ''}), '')
        with self.assertRaises(ValueError):
            ultrawide_patch({'ultrawide_res': '3000x1000'})
