# CHIMExIllustrisTNG
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX) <<-- placeholder -->

Intended to store code and mock catalogs and intensity maps related to Polzin, Newburgh, & Natarajan, in prep.

There are 3 (pre-beam convolution) intensity maps at each of z = 1 and 2 in the maps directory of this repository:
1. all galaxies (non-cluster galaxies and cluster galaxies)
2. mass-selected galaxy clusters
3. radius selected galaxy clusters

Additionally, there are pre-permutation catalogs corresponding to each of these intensity maps in the catalogs directory and code used in the creation and analysis of mock intensity maps and in the permutation of the volume/catalogs in the analysis directory. Ideally, this means that the code + catalogs are enough to do similar analyses without downloading tens of GBs of TNG300 group catalogs (though you will still need to download the [Molecular and atomic hydrogen (HI+H2) galaxy contents catalog](https://www.tng-project.org/data/docs/specifications/#sec5i) at z = 1 and 2, which are a total of ~4 GB).
