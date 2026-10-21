import os
import shutil
import hashlib
import json
import zipfile

supp_dir = 'supplemental_audit'
os.makedirs(supp_dir, exist_ok=True)

files_to_include = [
    # Step 1: Explicitly requested in prompt
    'verification/verifier.py',
    'retrieval/schema.py',
    'retrieval/bm25.py',
    'retrieval/dense.py',
    'conflict/schema.py',
    'stability/schema.py',
    'outputs/phase11_formal/.signals_cache.json',
    'requirements.txt',
    
    # Step 2: Additional local source modules required by recursive import tree
    'verification/schema.py',
    'verification/__init__.py',
    'utils/logging_setup.py',
    'utils/reproducibility.py',
    'utils/__init__.py',
    'retrieval/__init__.py',
    'conflict/__init__.py',
    'stability/__init__.py'
]

strictly_requested = [
    'verification/verifier.py',
    'retrieval/schema.py',
    'retrieval/bm25.py',
    'retrieval/dense.py',
    'conflict/schema.py',
    'stability/schema.py',
    'outputs/phase11_formal/.signals_cache.json',
    'requirements.txt'
]

def get_hash(fpath):
    sha = hashlib.sha256()
    with open(fpath, 'rb') as fp:
        while chunk := fp.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

copied_entries = []
missing_requested = []

for f in files_to_include:
    orig_norm = f.replace('\\', '/')
    if os.path.exists(orig_norm):
        dest = os.path.join(supp_dir, orig_norm)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(orig_norm, dest)
        size = os.path.getsize(dest)
        sha = get_hash(dest)
        orig_sha = get_hash(orig_norm)
        assert sha == orig_sha, f'Hash mismatch on copy: {orig_norm}'
        copied_entries.append({
            'original_relative_path': orig_norm,
            'supplemental_relative_path': dest.replace('\\', '/'),
            'file_size': size,
            'sha256': sha
        })
    else:
        if orig_norm in strictly_requested:
            missing_requested.append(orig_norm)

# Manifest creation
manifest_data = {
    'metadata': {
        'manifest_name': 'SUPPLEMENTAL_AUDIT_MANIFEST',
        'created_at': '2026-10-20T14:53:00Z',
        'purpose': 'Address remaining evidence gaps for Pre-Phase-14 Independent Technical Audit',
        'total_files_included': len(copied_entries),
        'missing_requested_files': missing_requested,
        'final_ppt_search': {
            'pattern': 'final_ppt*.pptx',
            'found': False,
            'matches': [],
            'note': 'final_ppt*.pptx does not exist in project repository. Only Phase 0 proposal Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx is present.'
        }
    },
    'files': copied_entries
}

manifest_path_in_supp = os.path.join(supp_dir, 'SUPPLEMENTAL_AUDIT_MANIFEST.json')
manifest_path_root = 'SUPPLEMENTAL_AUDIT_MANIFEST.json'

with open(manifest_path_in_supp, 'w', encoding='utf-8') as fp:
    json.dump(manifest_data, fp, indent=2)

with open(manifest_path_root, 'w', encoding='utf-8') as fp:
    json.dump(manifest_data, fp, indent=2)

print(f'Successfully copied {len(copied_entries)} files into {supp_dir}')
print(f'Manifest written to {manifest_path_in_supp} and {manifest_path_root}')

# Step 5: Create supplemental_pre_phase14_audit.zip
zip_filename = 'supplemental_pre_phase14_audit.zip'

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(supp_dir):
        for f in sorted(files):
            full_path = os.path.join(root, f)
            arcname = full_path.replace('\\', '/')
            zf.write(full_path, arcname=arcname)
            print(f'Added to zip: {arcname}')

print(f'\nZip file {zip_filename} created successfully. Size: {os.path.getsize(zip_filename):,} bytes.')

# Step 6: Reopen the completed ZIP and validate its actual contents
print('\n=== STEP 6: REOPENING AND VALIDATING ZIP CONTENTS ===')

with open(manifest_path_in_supp, 'r', encoding='utf-8') as fp:
    manifest = json.load(fp)

manifest_files = {entry['supplemental_relative_path']: entry for entry in manifest['files']}

regular_files_count = 0
hash_mismatches = []
validated_files = []

with zipfile.ZipFile(zip_filename, 'r') as zf:
    namelist = zf.namelist()
    print(f'Total entries in zip: {len(namelist)}')
    for name in namelist:
        info = zf.getinfo(name)
        if info.is_dir():
            continue
        regular_files_count += 1
        data = zf.read(name)
        file_sha = hashlib.sha256(data).hexdigest()
        
        # Check against disk
        if os.path.exists(name):
            with open(name, 'rb') as fp:
                disk_sha = hashlib.sha256(fp.read()).hexdigest()
            if file_sha != disk_sha:
                hash_mismatches.append(f'Disk mismatch: {name} (zip: {file_sha}, disk: {disk_sha})')
        else:
            hash_mismatches.append(f'File in zip not on disk: {name}')
            
        # Check against manifest if listed
        if name in manifest_files:
            m_sha = manifest_files[name]['sha256']
            if file_sha != m_sha:
                hash_mismatches.append(f'Manifest mismatch: {name} (zip: {file_sha}, manifest: {m_sha})')
            validated_files.append((name, len(data), file_sha))
        elif name.endswith('SUPPLEMENTAL_AUDIT_MANIFEST.json'):
            print(f'Validated manifest file in zip: {name} ({len(data)} bytes)')
        else:
            print(f'Extra file in zip not in manifest: {name}')

missing_list = manifest['metadata']['missing_requested_files']

print(f'\nValidation results:')
print(f'Number of regular files in ZIP: {regular_files_count}')
print(f'Number of files validated against manifest: {len(validated_files)}')
print(f'Missing requested files: {missing_list}')
print(f'Hash mismatches: {len(hash_mismatches)}')
if hash_mismatches:
    for m in hash_mismatches:
        print('  Mismatch:', m)
else:
    print('ALL HASHES MATCH 100% PERFECTLY!')
