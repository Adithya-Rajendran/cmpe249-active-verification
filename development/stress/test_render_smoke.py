import json
import tempfile
from pathlib import Path

from development.stress import render_smoke as S
from sim import arrange as A


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def run():
    old = S.ROOT
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        S.ROOT = root
        geom = root / 'geometry'
        geom.mkdir()
        output = root / 'renders'
        protocol = root / 'development/stress/protocol.json'
        builder = root / 'development/stress/build_cohort.py'
        builder.parent.mkdir(parents=True)
        builder.write_text('test fixture')
        write(protocol, {'study_id': 'invalid', 'status': 'development'})
        try:
            S.preflight(geom, output, root, root, root)
            raise AssertionError('invalid study was accepted')
        except ValueError as exc:
            assert 'study ID' in str(exc)
        write(protocol, {'study_id': S.STUDY, 'status': 'development'})
        write(geom / 'summary.json', {'shelf': {'passes': True}, 'tabletop': {'passes': True}})
        write(geom / 'provenance.json', {'protocol_sha256': A.sha256_file(protocol), 'builder_sha256': A.sha256_file(builder)})
        write(geom / 'cohort_manifest.json', {'records': [{'arrangement_id': i, 'arrangement_file': i + '.json'} for i in S.IDS]})
        write(geom / (S.IDS[0] + '.json'), {'seed': 100000})
        try:
            S.preflight(geom, output, root, root, root)
            raise AssertionError('reserved seed was accepted')
        except A.SeedError:
            pass
        assert not output.exists()
    S.ROOT = old
    print('PASS invalid study and reserved seed refused before output or GPU startup')


if __name__ == '__main__':
    run()
