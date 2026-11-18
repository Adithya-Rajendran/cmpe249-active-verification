"""Exercise the generated offline reviewer with the installed JavaScriptCore library."""
import csv
import ctypes as C
import ctypes.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from development.stress.prepare_inputs import annotation_pack, case_id, sha


class JavaScript:
    def __init__(self):
        name = ctypes.util.find_library('javascriptcoregtk-4.0')
        if not name:
            raise unittest.SkipTest('JavaScriptCore system library not available')
        self.lib = C.CDLL(name)
        definitions = {
            'JSGlobalContextCreate': ([C.c_void_p], C.c_void_p),
            'JSGlobalContextRelease': ([C.c_void_p], None),
            'JSStringCreateWithUTF8CString': ([C.c_char_p], C.c_void_p),
            'JSStringRelease': ([C.c_void_p], None),
            'JSEvaluateScript': ([C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.POINTER(C.c_void_p)], C.c_void_p),
            'JSValueToStringCopy': ([C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)], C.c_void_p),
            'JSStringGetMaximumUTF8CStringSize': ([C.c_void_p], C.c_size_t),
            'JSStringGetUTF8CString': ([C.c_void_p, C.c_void_p, C.c_size_t], C.c_size_t),
        }
        for name, (args, result) in definitions.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result
        self.ctx = self.lib.JSGlobalContextCreate(None)

    def close(self):
        self.lib.JSGlobalContextRelease(self.ctx)

    def evaluate(self, code):
        script = self.lib.JSStringCreateWithUTF8CString(code.encode())
        error = C.c_void_p()
        value = self.lib.JSEvaluateScript(self.ctx, script, None, None, 0, C.byref(error))
        self.lib.JSStringRelease(script)
        string = self.lib.JSValueToStringCopy(self.ctx, error.value or value, None)
        buf = C.create_string_buffer(self.lib.JSStringGetMaximumUTF8CStringSize(string))
        self.lib.JSStringGetUTF8CString(string, buf, len(buf)); self.lib.JSStringRelease(string)
        if error.value: raise AssertionError(buf.value.decode())
        return buf.value.decode()


class ReviewerTests(unittest.TestCase):
    def test_smoke_import_to_full_is_bound_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); folder = root / 'verifier/primary'; folder.mkdir(parents=True)
            (root / 'preparation.json').write_text('{}')
            jobs = []
            for number, level in enumerate(('D0', 'D6', 'D1')):
                ident = case_id({'environment': 'shelf', 'base_id': 'A', 'level_id': level}, 'P1', 'visible')
                for side in ('r', 'q'):
                    Image.new('RGB', (1200,675), (number, number, number)).save(folder / (ident + '_' + side + '.png'))
                jobs.append({'id': ident, 'reference': ident+'_r.png', 'query': ident+'_q.png', 'box': [1,2,3,4],
                             'reference_sha256': sha(folder/(ident+'_r.png')), 'query_sha256': sha(folder/(ident+'_q.png'))})
            annotation_pack(root, jobs)
            html = (root/'review/index.html').read_text()
            script = re.search(r'<script>(.*?)</script>', html, re.S).group(1)
            runtime = JavaScript(); self.addCleanup(runtime.close)
            runtime.evaluate('''var store={}; var localStorage={getItem:k=>store[k]||null,setItem:(k,v)=>store[k]=v};
              var counter={},ref={},query={},label={value:''},cue={value:''},reviewer={value:''},message={};
              var captured=''; var Blob=function(parts){captured=parts.join('')};
              var URL={createObjectURL:()=>''}; var document={createElement:()=>({click:()=>{}})};''')
            runtime.evaluate(script)
            with (root/'review/blank_labels.csv').open(newline='') as f:
                rows = list(csv.DictReader(f)); fields = list(rows[0])
            by_id = {row['id']:row for row in rows}
            first, second = by_id[jobs[0]['id']], by_id[jobs[1]['id']]
            first.update(label='PRESENT', cue='Visible "mark",\nsecond line', reviewer='Reviewer A')
            second.update(label='UNKNOWN', cue='Not enough detail', reviewer='Reviewer B')
            def csv_text(values):
                output=io.StringIO(newline=''); writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\r\n')
                writer.writeheader(); writer.writerows(values); return output.getvalue()
            def do_import(text):
                runtime.evaluate('importCSV({files:[{text:()=>Promise.resolve('+json.dumps(text)+')}],value:""});')
                return runtime.evaluate('message.textContent')
            # Smoke is a strict subset of full, in a different order.
            self.assertEqual(do_import(csv_text([second, first])), 'Imported 2 labels.')
            saved = json.loads(runtime.evaluate('JSON.stringify(saved)'))
            self.assertEqual(saved[first['id']]['cue'], 'Visible "mark",\nsecond line')
            self.assertEqual(saved[second['id']]['reviewer'], 'Reviewer B')
            runtime.evaluate('download()')
            exported = list(csv.DictReader(io.StringIO(runtime.evaluate('captured'), newline='')))
            exported_by_id = {row['id']: row for row in exported}
            self.assertEqual(exported_by_id[first['id']]['cue'], 'Visible "mark",\nsecond line')
            self.assertEqual(exported_by_id[first['id']]['query_sha256'], first['query_sha256'])
            before = runtime.evaluate('JSON.stringify(saved)')
            # A valid new row before a mismatched row must not be applied.
            third = dict(by_id[jobs[2]['id']], label='REMOVED', cue='Exposed', reviewer='Reviewer C')
            bad = dict(first, query_sha256='0'*64)
            self.assertTrue(do_import(csv_text([third,bad])).startswith('Import refused:'))
            self.assertEqual(runtime.evaluate('JSON.stringify(saved)'), before)
            for field in ('reference_sha256','box_sha256','input_size','id'):
                changed = dict(first); changed[field] = 'bad'
                self.assertTrue(do_import(csv_text([third,changed])).startswith('Import refused:'))
                self.assertEqual(runtime.evaluate('JSON.stringify(saved)'), before)
            conflict = dict(first, label='REMOVED')
            self.assertTrue(do_import(csv_text([third, conflict])).startswith('Import refused:'))
            self.assertEqual(runtime.evaluate('JSON.stringify(saved)'), before)
            self.assertTrue(do_import(csv_text([third, third])).startswith('Import refused:'))
            self.assertEqual(runtime.evaluate('JSON.stringify(saved)'), before)

if __name__ == '__main__': unittest.main()
