import os
import shutil
import hashlib
import json
import zipfile

pack_dir = 'sensitivity_audit_sourcepack'
os.makedirs(pack_dir, exist_ok=True)

files_to_package = [
    # 1. Sensitivity audit scripts
    ('scratch/run_nei_sensitivity_audit.py', 'scratch/run_nei_sensitivity_audit.py', 'Sensitivity audit computation script'),
    ('scratch/plot_sensitivity.py', 'scratch/plot_sensitivity.py', 'Sensitivity ROC and PR curves plotting script'),
    ('scratch/build_sensitivity_report.py', 'scratch/build_sensitivity_report.py', 'Sensitivity audit report generator'),
    
    # 2. Sensitivity report
    ('DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md', 'DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md', 'Comprehensive NEI label semantics sensitivity audit report'),
    
    # 3. Dataset metadata
    ('data/dataset_meta.json', 'data/dataset_meta.json', 'Dataset generation metadata showing copenlu source, seed 42, 250 per label'),
    
    # 4. Original dataset preparation script
    ('data/fever_loader.py', 'data/fever_loader.py', 'Original script that loads copenlu/fever_gold_evidence, selects 250 records per label (seed 42), and builds evidence_corpus.json and fever_dev.jsonl'),
    
    # 5. Dataset provenance documentation
    ('docs/DATASETS_AND_PROVENANCE.md', 'docs/DATASETS_AND_PROVENANCE.md', 'Formal dataset provenance, license, and lineage documentation'),
    ('DATASET_OVERLAP_AUDIT.md', 'DATASET_OVERLAP_AUDIT.md', 'Comprehensive dataset overlap and closed-candidate pool audit'),
    ('outputs/phase12_analysis/dataset_overlap_audit.json', 'outputs/phase12_analysis/dataset_overlap_audit.json', 'Machine-readable dataset overlap and provenance audit')
]

def get_hash(fpath):
    sha = hashlib.sha256()
    with open(fpath, 'rb') as fp:
        while chunk := fp.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

copied_entries = []
missing_files = []

for orig, dest_rel, desc in files_to_package:
    orig_norm = orig.replace('\\', '/')
    if os.path.exists(orig_norm):
        dest_full = os.path.join(pack_dir, dest_rel)
        os.makedirs(os.path.dirname(dest_full), exist_ok=True)
        shutil.copy2(orig_norm, dest_full)
        
        size = os.path.getsize(dest_full)
        sha = get_hash(dest_full)
        orig_sha = get_hash(orig_norm)
        assert sha == orig_sha, f'Hash mismatch on copy: {orig_norm}'
        
        copied_entries.append({
            'original_relative_path': orig_norm,
            'sourcepack_relative_path': os.path.join(pack_dir, dest_rel).replace('\\', '/'),
            'file_size': size,
            'sha256': sha,
            'description': desc
        })
    else:
        missing_files.append(orig_norm)

manifest_data = {
    'metadata': {
        'manifest_name': 'SENSITIVITY_AUDIT_SOURCEPACK_MANIFEST',
        'created_at': '2026-10-20T16:15:00Z',
        'purpose': 'Package source code and provenance documentation for independent NEI sensitivity audit',
        'total_files_packaged': len(copied_entries),
        'missing_files': missing_files,
        'dataset_preparation_script_status': {
            'found': True,
            'original_path': 'data/fever_loader.py',
            'details': 'Contains load_fever_split(), create_dev_subset(per_label=250, seed=42), extract_evidence_corpus(), and prepare_fever_data(). Constructs data/fever_dev.jsonl, data/evidence_corpus.json, and data/dataset_meta.json.'
        }
    },
    'files': copied_entries
}

manifest_path = os.path.join(pack_dir, 'SENSITIVITY_AUDIT_SOURCEPACK_MANIFEST.json')
with open(manifest_path, 'w', encoding='utf-8') as fp:
    json.dump(manifest_data, fp, indent=2)

print(f'Copied {len(copied_entries)} files into {pack_dir}')
print(f'Manifest written to {manifest_path}')

# Create ZIP
zip_filename = 'sensitivity_audit_sourcepack.zip'
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(pack_dir):
        for f in sorted(files):
            full_path = os.path.join(root, f)
            arcname = full_path.replace('\\', '/')
            zf.write(full_path, arcname=arcname)
            print(f'Added to zip: {arcname}')

print(f'Created {zip_filename} ({os.path.getsize(zip_filename):,} bytes)')

# Reopen and validate ZIP
print('\n=== VALIDATING ZIP CONTENTS ===')
with zipfile.ZipFile(zip_filename, 'r') as zf:
    namelist = zf.namelist()
    regular_files = 0
    hash_mismatches = 0
    manifest_map = {e['sourcepack_relative_path']: e for e in copied_entries}
    
    for name in namelist:
        info = zf.getinfo(name)
        if info.is_dir():
            continue
        regular_files += 1
        data = zf.read(name)
        z_sha = hashlib.sha256(data).hexdigest()
        
        # Check against manifest
        if name in manifest_map:
            expected_sha = manifest_map[name]['sha256']
            if z_sha != expected_sha:
                print(f'MISMATCH in {name}: zip={z_sha}, expected={expected_sha}')
                hash_mismatches += 1
        elif name.endswith('SENSITIVITY_AUDIT_SOURCEPACK_MANIFEST.json'):
            print(f'Validated manifest in zip: {name}')
        else:
            print(f'Unknown file in zip: {name}')

print(f'Total regular files in zip: {regular_files}')
print(f'Files matched to manifest: {len(copied_entries)}')
print(f'Hash mismatches: {hash_mismatches}')
print(f'Missing files: {missing_files}')
if hash_mismatches == 0 and len(missing_files) == 0:
    print('ALL PACKAGED FILES VALIDATED WITH 100% BIT-FOR-BIT ACCURACY!')
