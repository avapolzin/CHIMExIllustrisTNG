# CHIMExIllustrisTNG
[![arXiv](https://img.shields.io/badge/arXiv-2404.01382-b31b1b)](https://arxiv.org/abs/2404.01382)

Intended to store code and mock catalogs and intensity maps related to [Polzin et al. in press](https://arxiv.org/abs/2404.01382).

There are 3 (pre-beam convolution, beam-convolved, and beam-convolved + padded) intensity maps at each of z = 1 and 2 in the maps directory of this repository:
1. all galaxies (field galaxies and cluster galaxies)
2. mass-selected galaxy clusters
3. radius selected galaxy clusters

These maps, and the beams, are saved as .npy files, which can be read in the following way:
```python
import numpy as np
filename = np.load('filename.npy')
```

Additionally, there are pre-permutation catalogs corresponding to each of these intensity maps in the catalogs directory and code used in the creation and analysis of mock intensity maps and in the permutation of the volume/catalogs in the analysis directory. Ideally, this means that the code + catalogs are enough to do similar analyses without downloading tens of GBs of TNG300 group catalogs (though you will still need to download the [Molecular and atomic hydrogen (HI+H2) galaxy contents catalog](https://www.tng-project.org/data/docs/specifications/#sec5i) at z = 1 and 2, which are a total of \~4 GB).

We offer a guide to what changes should be made to the code for different use cases.

### How to modify code + catalogs to match your work:
- Since everything should be accessible in the target catalogs we provide here + the [Molecular and atomic hydrogen (HI+H2) galaxy contents catalog](https://www.tng-project.org/data/docs/specifications/#sec5i), the only reason you should need to use the TNG300 group catalogs in a repeated analysis are if you are changing the cluster selection criteria. If you are changing the HI-H<sub>2</sub> model, you can simply overwrite the pre-permutation catalogs with new M<sub>HI</sub> values from the "Molecular and atomic hydrogen galaxy contents catalog" and re-run the tiling/stacking. *If you are changing the cluster selection criteria, then you will need to modify `check_M200` and the condition on R200 in `chime_mock_catalog.py` and re-run `build_HI_cats`.*<
- If you are using this code/these catalogs with another experiment, you will need to feed `chime_mock_analysis.py` new arrays representing your beam, will want to modify the pixel resolution in `make_bins`, `tile`, and `make_stack_map`, swapping out 5.3 and 5.7 (arcmins) for the relevant values in your case, and will want to change the global "freq_width" to correspond to your instrument's spectral resolution. Similarly, the stacks being of shape (69, 65) is due to the angular resolution of the pixels, so you will also want to change amount of padding in `make_stack_map`, and the pad conditions in `stack` and `nsamp_stack`. (You should be able to find+replace all for the following values 5.3, 5.7 -- for the pixel resolution, 34, 32 -- for the padding on the stacks, and 69, 65 -- for the stack dimensions to easily align with another instrument's specifications.) At minimum, you will also want to recompute the beam FWHM and f<sub>nt</sub> (code at the end of `chime_mock_analysis.py`) to replace "sigma" in `single_gauss_0` and `single_gauss_1` and the global variables "fnt" and "fff" (the latter of which is also going to be catalog dependent), and you will likely need to recalibrate the mass recovery to suit your instrument and its beam.
- In any case, `chime_make_figures.py` should be used as a guide as and should *not* be run as a separate script as it relies on variables defined in `chime_mock_analysis.py`.

