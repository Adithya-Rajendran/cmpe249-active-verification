"""Validate the fixed development cohort before model use."""
from boundary import load_image

IDS={'shelf_dev_v3':['SD1000','SD1001','SD1002','SD1003','SD1004','SD1005'],
     'tabletop_dev49':['TD5011','TD5012','TD5013','TD5014','TD5015','TD5016']}

def validate(manifest,folder):
    if manifest.get('study_id')!='support-dev-pilot-v1': raise ValueError('wrong study identifier')
    if manifest.get('evidence_status') != 'SANITIZED_UNMARKED_V1': raise ValueError('evidence is not ready')
    rows=manifest['rows']; expected=set()
    for env,ids in IDS.items():
        for i,name in enumerate(ids):
            partition='calibration' if i%2==0 else 'pilot_evaluation'
            for stop in ('P1','P3','P5'):
                for phase in ('visible','occluded','removed'): expected.add((env,name,partition,stop,phase))
    actual=[(r['environment'],r['arrangement'],r['partition'],r['stop'],r['phase']) for r in rows]
    if len(rows)!=108 or len(set(actual))!=108 or set(actual)!=expected or {r['id'] for r in rows}!=set(range(108)):
        raise ValueError('unexpected cohort, duplicate or missing record')
    for r in rows:
        if r['support'] not in {'PRESENT','REMOVED','UNKNOWN','DISPUTED'} or r['reviewer_type']!='AI': raise ValueError('missing AI review')
        for k in ('reference','query'): load_image(folder,r[k])
