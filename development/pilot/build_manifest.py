"""Initialize the fixed development manifest with evidence and review pending."""
import json
from pathlib import Path
OUT=Path('runs/2026-11-16-support-audit-v1')
def main():
    ledger=json.loads((OUT/'ledger.json').read_text()); images=json.loads((OUT/'prepared_inputs.json').read_text())
    result=[]
    for i,(row,img) in enumerate(zip(ledger,images,strict=True)):
        if img['record_index']!=i: raise ValueError('prepared input ordering mismatch')
        result.append({'id':i,'arrangement':row['arrangement'],'partition':row['partition'],
          'environment':row['environment'],'stop':row['stop'],'phase':row['phase'],
          'physical':row['physical_state'].upper(),'support':'PENDING','reviewer_type':'PENDING',
          'target_description':row['target'].replace('_',' '),'reference':img['reference'],'query':img['query'],
          'valid':None,'quality':None,'match':None,'obstruction':None,'evidence_failure':None})
    (OUT/'pilot_manifest.json').write_text(json.dumps({'study_id':'support-dev-pilot-v1',
      'evidence_status':'PENDING','rows':result},indent=2)+'\n')
    print(len(result),'records; review and evidence pending')
if __name__=='__main__':main()
