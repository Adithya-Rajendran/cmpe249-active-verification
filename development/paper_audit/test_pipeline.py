import json,tempfile
from pathlib import Path
from PIL import Image,ImageDraw
from unittest.mock import patch
from development.paper_audit.inputs_build import build,sha
from development.paper_audit.inputs_validate import validate
from development.paper_audit.evaluate_results import evaluate
from development.paper_audit.evaluate_calibration import evaluate as evaluate_calibration
with tempfile.TemporaryDirectory() as d:
 root=Path(d); img=root/'rgb.png'; m=root/'mask.png'
 Image.new('RGB',(2304,1296),(3,4,5)).save(img)
 im=Image.new('L',(2304,1296)); ImageDraw.Draw(im).rectangle((900,400,1000,500),fill=255);im.save(m)
 empty=root/'empty.png'; Image.new('L',(2304,1296)).save(empty)
 entry=lambda p:{'path':str(p),'sha256':sha(p)}
 labels=['clear_present','clear_removed','placed_present','placed_removed']
 images={s:{'rgb':entry(img),'target_mask':entry(empty if s.endswith('removed') else m),'blocker_mask':entry(m)} for s in labels}
 obsdir=root/'A'/'D0';obsdir.mkdir(parents=True)
 observed=obsdir/'observed.json'; observed.write_text('{}')
 row={'observed':entry(observed),'engine':'isaac','arrangement':'A','environment':'shelf','level':'D6','stop':'P1','reference':{'rgb':entry(img),'target_mask':entry(m)},'images':images,'isolated_target_mask':entry(m)}
 manifest=root/'manifest.json';manifest.write_text(json.dumps({'rows':[row]})); packet=root/'packet';ledger=root/'private'/'ledger.json'
 build([manifest],packet,ledger)
 cases=json.loads((packet/'cases.json').read_text()); scoring=root/'scoring';scoring.mkdir()
 (scoring/'binding.json').write_text(json.dumps({'cases_sha256':sha(packet/'cases.json')}))
 (scoring/'records.jsonl').write_text(''.join(json.dumps({'id':c['id'],'answer':'UNKNOWN','reference_sha256':c['reference_sha256'],'query_sha256':c['query_sha256'],'box_xyxy':c['box_xyxy']})+'\n' for c in cases))
 receipt=root/'receipt.json'; receipt.write_text(json.dumps({'manifest':entry(manifest),'sources':{},'inputs_sha256':{str(observed):sha(observed)},'rows':[{'base_id':'A','level':'D0','failures':[]}]}))
 evaluate(packet,ledger,{'baseline_200':str(scoring/'records.jsonl'),'geometry':str(scoring/'records.jsonl')},root/'evaluation',[receipt],repetitions=20)
 try:
  validate(packet,ledger,[receipt],root/'must-not-release.json')
  raise AssertionError('incomplete cohort was released')
 except ValueError as exc:
  assert 'incomplete declared cohort' in str(exc)
 assert not (root/'must-not-release.json').exists()
 summaries=json.loads((root/'evaluation'/'summary.json').read_text());cert=json.loads((root/'evaluation'/'certificates.json').read_text())['pairs']
 assert len(cert)==8 and all(c['complete_input_equal'] for c in cert)
 assert next(x for x in summaries['groups'] if x['method']=='baseline_200')['coverage']==0
 criteria=json.loads((root/'evaluation'/'criteria.json').read_text())
 assert not criteria['tradeoff_free_improvement_supported'] and not criteria['engines'][0]['strict_primary_gain']
 cal_images=dict(images)
 for state in labels:
  for prefix in ('repeat1_','repeat2_','distractor_hidden_'): cal_images[prefix+state]=images[state]
 cal_images['distractor_hidden_clear_present']=dict(images['clear_present'],target_mask=entry(empty))
 calrow=dict(row,level='D0',camera={'pose':1},nominal_camera={'pose':1},images=cal_images,observed=entry(observed))
 cm=root/'cal.json';cm.write_text(json.dumps({'rows':[calrow]}))
 receipt=root/'receipt.json'; receipt.write_text(json.dumps({'manifest':entry(cm),'sources':{},'inputs_sha256':{str(observed):sha(observed)},'rows':[{'base_id':'A','level':'D0','failures':[]}]}))
 with patch('development.paper_audit.evaluate_calibration.verify_states',return_value=[{'state_geometry_matches':True}]):
  evaluate_calibration([cm],root/'calibration',[receipt])
 cal=json.loads((root/'calibration'/'calibration.json').read_text())
 assert not cal['witness_main_endpoint_permitted'] and len(cal['rows'])==2
 assert all(len(r['negatives'])==6 for r in cal['rows'])
 assert cal['rows'][0]['negatives'][2]['mask_changes']['target_mask']['native_removed']>0
 print('PASS: synthetic packet, predictions, pairs, certificates, summaries and calibration failure gate')
