import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image, PngImagePlugin
from boundary import load_image, make_messages

class BoundaryTests(unittest.TestCase):
    def test_metadata_and_filename_sentinel(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'REMOVED_SECRET_PAIR.png'; im=Image.new('RGB',(1200,675),'white')
            meta=PngImagePlugin.PngInfo();meta.add_text('cmpe249','REMOVED_SECRET_PAIR')
            im.save(p,pnginfo=meta);item={'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            with self.assertRaises(ValueError): load_image(d,item)
            im.save(p);item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            image=load_image(d,item); self.assertFalse(image.info)
            msg=make_messages('mug','Question. ')
            self.assertNotIn('REMOVED_SECRET_PAIR',repr(msg))
            self.assertNotIn(str(p),repr(msg))
    def test_injected_description_rejected(self):
        with self.assertRaises(ValueError): make_messages('mug REMOVED_SECRET_PAIR','Question. ')

if __name__=='__main__': unittest.main()
