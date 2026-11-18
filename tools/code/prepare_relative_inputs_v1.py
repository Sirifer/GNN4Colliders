#!/usr/bin/env python3
"""Build separate, slim ROOT inputs and configs; never modify source samples.
All momenta entering the model are divided by m_llg. This removes the
explicit energy scale, but does NOT establish statistical mass independence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import uproot
import yaml

BASE = Path(__file__).resolve().parents[2]
NODES = [f'jet{i}' for i in range(6)] + [f'bjet{i}' for i in range(2)] + [f'lep{i}' for i in range(6)] + ['photon', 'MET']
META = ['event', 'run', 'lumi', 'photon_eta_fold', 'weight_lumi', 'weight_lumi_avg', 'source_tree_tag', 'm_llg']
GLOBALS = ['rel_n_jets', 'rel_n_bjets', 'rel_n_leptons', 'rel_Z_pt_over_mllg', 'rel_dr_lead_gamma', 'rel_dr_sublead_gamma']

def required():
    keys = set(META + ['n_jets', 'n_bjets', 'n_leptons', 'Z_pt', 'photon_id_mva'])
    for obj in NODES:
        keys.update([obj+'_pt', obj+'_phi'])
        if obj != 'MET':
            keys.add(obj+'_eta')
        if obj.startswith(('jet', 'bjet')):
            keys.add(obj+'_hzg_btag_score')
        if obj.startswith('lep'):
            keys.add(obj+'_pdgId')
    for obj in ['lepton_lead', 'lepton_sublead']:
        keys.update([obj+'_eta', obj+'_phi', obj+'_pt'])
    return sorted(keys)

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

def transform(a):
    m = a['m_llg']
    if not np.all(np.isfinite(m) & (m > 0)):
        raise ValueError('Nonpositive/nonfinite m_llg; no events silently discarded')
    out = {k: a[k] for k in META}
    for key in ['photon_pt', 'lepton_lead_pt', 'lepton_sublead_pt']:
        if not np.all(np.isfinite(a[key]) & (a[key] > 0)):
            raise ValueError('Invalid candidate '+key)
    for obj in NODES:
        pt = a[obj+'_pt']
        if not np.all(np.isfinite(pt)):
            raise ValueError('Nonfinite '+obj+'_pt')
        valid = pt > 0
        eta = np.zeros_like(m) if obj == 'MET' else a[obj+'_eta']
        phi = a[obj+'_phi']
        if np.any(valid & (~np.isfinite(eta) | ~np.isfinite(phi) | (np.abs(eta) > 50) | (np.abs(phi) > np.pi+1e-5))):
            raise ValueError('Invalid angles for present '+obj)
        # Subtract the same reference also for pseudo-eta(MET)=0, preserving edges.
        fields = {'pt_over_mllg': pt/m, 'deta_gamma': eta-a['photon_eta'], 'dphi_gamma': wrap(phi-a['photon_phi'])}
        if obj.startswith(('jet', 'bjet')):
            fields['id'] = a[obj+'_hzg_btag_score']
        elif obj == 'photon':
            fields['id'] = a['photon_id_mva']
        else:
            fields['id'] = np.zeros_like(m)
        fields['pdgid'] = a[obj+'_pdgId'] if obj.startswith('lep') else np.full_like(m, 22 if obj == 'photon' else 0)
        for name, value in fields.items():
            out[f'rel_{obj}_{name}'] = np.where(valid, value, 0).astype('float32')
    for key in ['n_jets', 'n_bjets', 'n_leptons']:
        if np.any(a[key] < 0):
            raise ValueError('Invalid multiplicity '+key)
        out['rel_'+key] = a[key].astype('float32')
    if np.any(a['Z_pt'] < 0):
        raise ValueError('Invalid Z_pt')
    out['rel_Z_pt_over_mllg'] = (a['Z_pt']/m).astype('float32')
    for obj, name in [('lepton_lead', 'lead'), ('lepton_sublead', 'sublead')]:
        eta, phi = a[obj+'_eta'], a[obj+'_phi']
        if np.any((np.abs(eta) > 50) | (np.abs(phi) > np.pi+1e-5)):
            raise ValueError('Invalid candidate angles')
        out[f'rel_dr_{name}_gamma'] = np.hypot(eta-a['photon_eta'], wrap(phi-a['photon_phi'])).astype('float32')
    for key, val in out.items():
        if key.startswith('rel_') and not np.all(np.isfinite(val)):
            raise ValueError('Nonfinite output '+key)
    return out

def make_configs(output, config_dir):
    config_dir.mkdir(parents=True, exist_ok=True)
    for fold in [0, 1]:
        src = BASE/f'root_gnn_dgl/configs/hzg/ttH_vs_other_run3_v2_fold{fold}.yaml'
        cfg = yaml.safe_load(src.read_text())
        cfg['Training_Name'] = f'hzg_ttH_relative_v1_fold{fold}'
        cfg['Training_Directory'] = str(BASE/f'trainings/hzg/ttH_relative_v1_fold{fold}')
        cfg['Model']['args']['in_size'] = 6
        for ds in cfg['Datasets'].values():
            args = ds['args']
            args['raw_dir'] = str(output)
            args['save_dir'] = str(BASE/'graphs/hzg/ttH_relative_v1')
            args['node_branch_names'] = [[f'rel_{obj}_{feat}' for obj in NODES] for feat in ['pt_over_mllg', 'deta_gamma', 'dphi_gamma', 'id', 'pdgid']] + ['NODE_TYPE']
            args['node_feature_scales'] = [1, 1, 1, 1, 0.1, 1]
            args['global_features'] = GLOBALS
            args['tracking_info'] = list(dict.fromkeys(args.get('tracking_info', []) + ['m_llg']))
        path = config_dir/f'ttH_relative_v1_fold{fold}.yaml'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=BASE/'root_p2root_v2_merged')
    p.add_argument('--output-dir', type=Path, default=BASE/'root_p2root_v2_relative_v1')
    p.add_argument('--config-dir', type=Path, default=BASE/'tools/code/relative_v1_configs')
    p.add_argument('--step-size', type=int, default=50000)
    opt = p.parse_args()
    baseline = yaml.safe_load((BASE/'root_gnn_dgl/configs/hzg/ttH_vs_other_run3_v2_fold0.yaml').read_text())
    names = sorted({name for ds in baseline['Datasets'].values() for name in ds['args']['file_names']})
    files = [opt.input_dir/name for name in names]
    if not files:
        raise ValueError('No input ROOT files')
    if opt.output_dir.exists():
        raise FileExistsError('Use a new output directory: '+str(opt.output_dir))
    # Check every schema before writing any samples.
    for src in files:
        with uproot.open(src) as f:
            missing = set(required()) - set(f['merged_SR_Sideband'].keys())
            if missing:
                raise ValueError(f'{src}: missing {sorted(missing)}')
    opt.output_dir.mkdir(parents=True)
    report = []
    for src in files:
        dest = opt.output_dir/src.relative_to(opt.input_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.root.partial')
        n = 0
        with uproot.open(src) as f, uproot.recreate(tmp) as g:
            t = f['merged_SR_Sideband']
            prototype = transform(t.arrays(required(), entry_stop=0, library='np'))
            target = g.mktree('merged_SR_Sideband', {k: v.dtype for k,v in prototype.items()})
            for a in t.iterate(required(), step_size=opt.step_size, library='np'):
                b = transform(a)
                target.extend(b)
                n += len(b['m_llg'])
            assert n == t.num_entries
        with uproot.open(tmp) as check:
            assert check['merged_SR_Sideband'].num_entries == n
        tmp.rename(dest)
        report.append({'file': str(src.relative_to(opt.input_dir)), 'entries': n})
        print(f'{len(report)}/{len(files)} {dest.name}: {n}', flush=True)
    make_configs(opt.output_dir.resolve(), opt.config_dir)
    (opt.output_dir/'manifest.json').write_text(json.dumps({'source': str(opt.input_dir.resolve()), 'files': report, 'total_entries': sum(r['entries'] for r in report), 'model_globals': GLOBALS, 'mass_is_tracking_only': True}, indent=2))

if __name__ == '__main__':
    main()
