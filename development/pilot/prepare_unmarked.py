"""Prepare the gate's unmarked copies of the same resized reference RGB."""
import json
from pathlib import Path
from PIL import Image
P=Path('runs/2026-11-16-support-audit-v1')
def main():
    ledger=json.loads((P/'ledger.json').read_text()); images=json.loads((P/'prepared_inputs.json').read_text())
    out=P/'gate_references';out.mkdir(exist_ok=True); seen=set()
    for r,im in zip(ledger,images,strict=True):
        name=im['reference']['name']
        if name in seen: continue
        with Image.open(r['source_paths']['reference']) as src:
            clean=Image.new('RGB',src.size);clean.paste(src.convert('RGB'))
        clean.resize((1200,675),Image.LANCZOS).save(out/name)
        seen.add(name)
    print(len(seen),'unmarked references')
if __name__=='__main__':main()
