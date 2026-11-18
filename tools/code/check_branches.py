import uproot, numpy as np

# 信号 ttH
f_sig = uproot.open('/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2/Signal/ttH/2018.root')
t_sig = f_sig['SR__ttHhad;1']
sig_branches = set(t_sig.keys())

# 背景 TT (Run3)
f_bkg = uproot.open('/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2/Bkg_MC/TT/2022postEE.root')
t_bkg = f_bkg['SR__ttHhad;1']
bkg_branches = set(t_bkg.keys())

print('Signal branches: %d' % len(sig_branches))
print('Bkg branches:    %d' % len(bkg_branches))
print()
only_sig = sorted(sig_branches - bkg_branches)
only_bkg = sorted(bkg_branches - sig_branches)
print('In signal but NOT in bkg (%d):' % len(only_sig))
for b in only_sig:
    print('  ', b)
print()
print('In bkg but NOT in signal (%d):' % len(only_bkg))
for b in only_bkg:
    print('  ', b)
print()

# -------------------------------------------------------------------
# 填充率分析：各 object slot 的非零/非NaN比例
# 用 ttH SR__ttHhad (多事件的 region)
# -------------------------------------------------------------------
regions = ['ttHhad', 'ttHlep', 'untag', 'VH3l', 'VHpTmiss']
all_trees = ['SR__%s' % r for r in regions] + ['Sideband__%s' % r for r in regions]

import os

def load_branch(root_dir, branch, max_files=3):
    arrays = []
    files = sorted([f for f in os.listdir(root_dir) if f.endswith('.root')])[:max_files]
    for fname in files:
        try:
            f = uproot.open(os.path.join(root_dir, fname))
            for tr in all_trees:
                if tr + ';1' in f:
                    t = f[tr + ';1']
                    if t.num_entries > 0 and branch in t.keys():
                        arrays.append(t[branch].array(library='np'))
        except:
            pass
    return np.concatenate(arrays) if arrays else np.array([])

# object slots to check
slots = {
    'leptons': ['lep0_pt', 'lep1_pt', 'lep2_pt', 'lep3_pt', 'lep4_pt', 'lep5_pt'],
    'jets':    ['jet0_pt', 'jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet5_pt'],
    'bjets':   ['bjet0_pt', 'bjet1_pt'],
    'photons': ['pho0_pt', 'pho1_pt', 'pho2_pt'],
}

print('=' * 55)
print('Object slot fill rate (% events with pt > 0)')
print('Source: ttH signal, first 3 ROOT files, all 5 regions+SB')
print('=' * 55)

sig_dir = '/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2/Signal/ttH'
for group, branches in slots.items():
    print('\n[%s]' % group)
    for br in branches:
        arr = load_branch(sig_dir, br, max_files=3)
        if len(arr) == 0:
            print('  %-20s  not found' % br)
        else:
            fill = np.mean(arr > 0) * 100
            print('  %-20s  fill=%.1f%%  (N=%d)' % (br, fill, len(arr)))

# 同样对背景看一个
print()
print('=' * 55)
print('Same check on TT background')
print('=' * 55)
bkg_dir = '/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2/Bkg_MC/TT'
for group, branches in slots.items():
    print('\n[%s]' % group)
    for br in branches:
        arr = load_branch(bkg_dir, br, max_files=3)
        if len(arr) == 0:
            print('  %-20s  not found' % br)
        else:
            fill = np.mean(arr > 0) * 100
            print('  %-20s  fill=%.1f%%  (N=%d)' % (br, fill, len(arr)))
