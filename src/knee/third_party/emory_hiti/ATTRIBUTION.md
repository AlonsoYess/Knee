# Emory-HITI knee-crop core

Source: https://github.com/Emory-HITI/knee-crop/tree/c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93

Copyright (c) 2025 HITI-LAB. MIT license: [LICENSE](LICENSE).

The pinned files `config.py` and `pipeline.py` preserve the upstream
configuration and functions used for unilateral cropping. The pipeline copy
contains upstream functions from lines 112–436 only; unrelated CLI, image
loading, multiprocessing and writing code were omitted. Imports were adapted
to this package. One optional `return_geometry` argument was added to
`process_knee_side` to expose the exact requested box; its default return
and signal calculations remain unchanged. Synthetic tests compare its default
and instrumented image outputs. The project-specific adapter
`knee.roi_mcr004` adds DICOM normalization, inverse-coordinate tracking,
outward rounding, native-pixel extraction and conservative abstention.

This is an adaptation, not a claim of reproducing the published study or of
clinical anatomical accuracy. MCR-2026-004 governs its evaluation.
