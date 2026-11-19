"""Run the unchanged stress evidence extractor with cross-simulator bindings."""
import argparse
import os
import socket
from pathlib import Path
from development.cross_sim.scoring_prepare import PROTOCOL, STUDY, verify_release, check_fixed_protocol
from development.stress.prepare_inputs import load, sha
from development.stress import run_evidence


def verify_offline_guard(guard_record):
    guard = load(guard_record)
    expected_guard = Path(__file__).resolve().parents[1] / 'stress/offline_exec.py'
    if (guard.get('guard_sha256') != sha(expected_guard) or
            Path(guard.get('script', '')).resolve() != Path(__file__).resolve() or
            guard.get('clip_sha256') != '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af' or
            any(os.environ.get(k) != v for k, v in {'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'YOLO_AUTOINSTALL':'false'}.items())):
        raise ValueError('offline guard binding mismatch')
    for family, address in ((socket.AF_INET, ('127.0.0.1', 9)), (socket.AF_INET6, ('::1', 9))):
        try:
            with socket.socket(family) as sock:
                sock.connect(address)
        except PermissionError:
            continue
        except OSError as exc:
            raise ValueError('socket guard is not active') from exc
        raise ValueError('socket guard is not active')
    try:
        socket.getaddrinfo('localhost', 80)
    except PermissionError:
        return guard
    raise ValueError('DNS guard is not active')


def run(prepared, out, control, device, guard_record, synthetic_self_test=False):
    guard = verify_offline_guard(guard_record)
    check_fixed_protocol()
    prepared = Path(prepared).resolve()
    pre = load(prepared / 'preparation.json')
    if synthetic_self_test:
        if pre.get('scope') != 'synthetic-self-test' or pre.get('count') != 1: raise ValueError('invalid synthetic self-test')
    elif pre.get('scope') != 'pilot' or pre.get('count') != 1200:
        raise ValueError('full pilot preparation required')
    for key in ('qualification_release', 'pilot_release'):
        if sha(pre[key]) != pre[key + '_sha256']: raise ValueError('release changed')
    verify_release(pre['qualification_release'], 'qualification-accepted', pre['qualification_manifest'])
    verify_release(pre['pilot_release'], 'pilot-inputs-accepted', pre['cohort_manifest'])
    out = Path(out).resolve()
    if out.exists() and not (out / 'cross_binding.json').is_file():
        raise ValueError('missing cross-simulator guard binding')
    out.mkdir(parents=True, exist_ok=True)
    binding_path = out / 'cross_binding.json'
    cross_binding = {'runner_sha256': sha(__file__), 'guard_sha256': guard['guard_sha256'], 'clip_sha256': guard['clip_sha256'], 'protocol_sha256': sha(PROTOCOL), 'control': control, 'synthetic_self_test': synthetic_self_test}
    if binding_path.exists():
        if load(binding_path) != cross_binding: raise ValueError('cross-simulator guard binding changed')
    else:
        binding_path.write_text(__import__('json').dumps(cross_binding, indent=2) + '\n')
    guard_copy = out / ('guard-' + sha(guard_record) + '.json')
    if not guard_copy.exists(): guard_copy.write_bytes(Path(guard_record).read_bytes())
    run_evidence.PROTOCOL = PROTOCOL
    run_evidence.STUDY = STUDY
    run_evidence.run(prepared, out / 'evidence', control, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--control', choices=['primary', 'relaxed_guard', 'native_normalized_guard'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--guard-record', required=True)
    parser.add_argument('--synthetic-self-test', action='store_true')
    args = parser.parse_args()
    run(args.prepared, args.out, args.control, args.device, args.guard_record, args.synthetic_self_test)
