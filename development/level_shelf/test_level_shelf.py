import tempfile
import unittest
from pathlib import Path
from PIL import Image
from development.level_shelf.evaluate import pixel,OLD_STOPS,HISTORICAL
from development.level_shelf.prepare import ENGINES,LAYOUTS,STOPS,STATES


class TestLevelShelf(unittest.TestCase):
    def test_declared_inventory_and_historical_mapping(self):
        self.assertEqual(len(ENGINES)*len(LAYOUTS)*len(STOPS)*len(STATES),320)
        self.assertEqual(OLD_STOPS,{'P1':'P4','P2':'P4','P4':'P5','P5':'P5'})
        self.assertEqual(len(HISTORICAL),4)
        self.assertEqual(len({a for a,_ in HISTORICAL}),3)

    def test_pixel_equality_uses_decoded_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.png';b=Path(directory)/'b.png'
            Image.new('RGB',(9,5),(2,3,4)).save(a,compress_level=0)
            Image.new('RGB',(9,5),(2,3,4)).save(b,compress_level=9)
            self.assertEqual(pixel(a),pixel(b))
            Image.new('RGB',(5,9),(2,3,4)).save(b)
            self.assertNotEqual(pixel(a),pixel(b))

if __name__=='__main__':unittest.main()
