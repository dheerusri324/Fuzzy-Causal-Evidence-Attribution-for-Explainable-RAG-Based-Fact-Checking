# Historical Audit Bundle (Archived Snapshot)

> [!WARNING]
> **FROZEN ARCHIVE NOTICE**: This directory contains an archived historical snapshot of audit assets and source files generated on **2026-10-20**. All contents in `audit_bundle/artifacts/` and `audit_bundle/source_files/` are cryptographically sealed and tracked by SHA-256 hashes in `audit_bundle/integrity_manifest.json`.
>
> **DO NOT MODIFY OR DELETE** files inside this directory, as altering them will invalidate the cryptographic integrity manifest and historical audit trail.

---

## Authoritative Documentation & Corrected Findings

For current, authoritative documentation, corrected citation records, and label-semantics sensitivity evaluations, refer to the root repository:

1. **Root Readme**: [`../README.md`](../README.md)
   - Contains the authoritative system architecture, current codebase organization, and corrected citation metadata.
2. **Post-Phase-13 Errata**: [`../PRE_PHASE_14_AUDIT_ERRATA_V4.md`](../PRE_PHASE_14_AUDIT_ERRATA_V4.md)
   - Resolves all historical reporting errata, including the 886 fuzzy fallback vs. 20 legitimate 0.5 defuzzifications in Experiment 2 (Phase 11).
3. **Dataset Label Semantics Sensitivity Audit**: [`../DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md`](../DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md)
   - Establishes the distinction between the 296 genuine human-annotated gold evidence passages (154 SUPPORTS + 142 REFUTES) and 147 candidate proxy passages (from 166 NOT_ENOUGH_INFO claims in `copenlu/fever_gold_evidence`).
   - Details Protocol A (500 claims; NEI treated as negative distractors; $\Delta \text{AUROC} = +0.0898$, $p = 0.0020$) and Protocol B (334 verifiable claims; NEI claims excluded; $\Delta \text{AUROC} = +0.0827$, $p = 0.0020$).
4. **Foundation Paper Citation Correction**:
   - The reference IEEE publication details are:
     - **Authors**: Jean-Flavien Bussotti, Andrea Baraldi, Francesco Guerra, and Paolo Papotti
     - **Title**: *Explaining the Role of Evidence in Data-Driven Fact-Checking*
     - **Journal**: *IEEE Access*, **vol. 14, pp. 127752–127765, 2026**
     - **DOI**: [10.1109/ACCESS.2026.3719082](https://doi.org/10.1109/ACCESS.2026.3719082)
     *(Note: Earlier historical documents in this archive cited vol. 12, pp. 37190–37204, which was a preliminary placeholder).*
