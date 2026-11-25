"""Integrity regressions for the offline evidence renderer, using exact capture fixtures."""
import copy
import gzip
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('optical_gallery', HERE/'build_optical_gallery.py')
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


class OpticalGalleryIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.item = json.loads((HERE/'prototype-optical-decisions.json').read_text())[0]
        folder = gallery.CAPTURES / cls.item['observation']
        cls.snapshot = json.loads((folder/'snapshot.json').read_text())
        cls.sha = gallery.helpers.digest((folder/'native.png').read_bytes())

    def check(self, item):
        return gallery.validate_decision(item, self.snapshot, self.sha)

    def test_actual_source_passes(self):
        actual = self.check(self.item)
        self.assertFalse(actual['physical_primary_proven'])
        self.assertEqual(sum(len(actual[k]) for k in ('selected', 'rejected', 'ambiguous')),
                         actual['count'])

    def test_rejects_photo_substitution(self):
        item = copy.deepcopy(self.item)
        item['png_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.check(item)

    def test_rejects_other_exposure_same_calibration(self):
        item = copy.deepcopy(self.item)
        item['cloud']['sequence'] += 1
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.check(item)

    def test_rejects_cross_table_index(self):
        item = copy.deepcopy(self.item)
        item['cloud']['reflection']['selected_candidate_indices'].append(1_000_000)
        with self.assertRaisesRegex(ValueError, 'candidate-table index'):
            self.check(item)

    def test_rejects_invented_selected_uv(self):
        item = copy.deepcopy(self.item)
        item['cloud']['uv'][0][0] += .1
        with self.assertRaisesRegex(ValueError, 'Selected cloud UV differs'):
            self.check(item)

    def test_gap_requires_exact_photo_hash(self):
        with self.assertRaisesRegex(ValueError, 'photo hash'):
            gallery.gap_for([{'observation': 6, 'native_png_sha256': '0'*64}], 6, self.sha)

    def test_compressed_gallery_round_trip(self):
        import base64
        import re
        html = (HERE/'optical-comparison.html').read_text()
        encoded = re.search(r'<script id="evidence"[^>]*>(.*?)</script>', html).group(1)
        data = json.loads(gzip.decompress(base64.b64decode(encoded)))
        self.assertEqual([15, 6, 53, 8, 47], [c['index'] for c in data['cards']])
        self.assertIsInstance(data['prototype'], bool)
        self.assertEqual(data['display_gamma_before_jpeg'], .5)
        if not data['prototype']:
            self.assertTrue(all(c['gap'][0]['found'] is False for c in data['cards']))
            self.assertTrue(all(c['gap'][0]['optical_support']['qualified'] is False
                                for c in data['cards']))
        self.assertLess(len(html.encode()), 1_000_000)


if __name__ == '__main__':
    unittest.main()
