from paths import ROOT
import struct
import unittest
from patches import (ULTRAWIDE_PATCHES, ultrawide_patch, ultrawide_writes, compile_patches,
                     EBOOT_BASE, SCENE_WIDTH, SCENE_HEIGHT, UI_WIDTH, UI_HEIGHT)
from test_patches import immediate


class UltrawideTests(unittest.TestCase):
    def test_complete_native_patch_with_retail_heaps(self):
        for size, name in ULTRAWIDE_PATCHES.items():
            with self.subTest(size=size):
                width, height = map(int, size.split('x'))
                self.assertEqual(ultrawide_patch({'ultrawide_res': size}), name)
                xml = ROOT / 'patches/Bloodborne.xml'
                writes = ultrawide_writes(xml, name, '01.09', [(0, 0x6000000)])
                full = compile_patches(xml, [name], '01.09', [(0, 0x6000000)])
                heaps = {0x04B36DDA - EBOOT_BASE, 0x04B36DE2 - EBOOT_BASE}
                self.assertEqual(writes, [w for w in full if w[0] not in heaps])
                for addr in heaps:
                    self.assertFalse(any(a < addr + 2 and a + len(d) > addr for a, d in writes))
                # Regression: removing these MOV/NOPs while keeping the fallback
                # globals silently allocated 640x360 and broke the entire UI.
                for address, expected in ((SCENE_WIDTH, width), (SCENE_HEIGHT, height),
                                          (UI_WIDTH, width), (UI_HEIGHT, height)):
                    self.assertEqual(immediate(writes, address)[1], expected)
                camera = dict(writes)[0x0183A35D - EBOOT_BASE]
                self.assertAlmostEqual(struct.unpack('<f', camera)[0], width / height, places=5)

    def test_off_and_invalid_resolution(self):
        self.assertEqual(ultrawide_patch({}), '')
        self.assertEqual(ultrawide_patch({'ultrawide_res': ''}), '')
        with self.assertRaises(ValueError):
            ultrawide_patch({'ultrawide_res': '3000x1000'})
