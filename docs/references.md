# References supporting image formation, corruption, and matching choices

The numbering below is stable for reuse in the hackathon presentation. These references support engineering approximations and numerical methods; they do not validate the synthetic generator as a physical SEM simulator.

## [1] Reimer, L.

Ludwig Reimer, *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*, 2nd edition, Springer, 1998. DOI: [10.1007/978-3-540-38967-5](https://doi.org/10.1007/978-3-540-38967-5).

Supports:

- treating electron/probe imaging as an image-formation process rather than ordinary photographic noise;
- using shot-noise language cautiously for a synthetic detector-count approximation;
- documenting that the repository's charging and scanline terms are approximations, not a calibrated SEM model.

## [2] Goldstein, J. I., Newbury, D. E., Michael, J. R., Ritchie, N. W. M., Scott, J. H. J., and Joy, D. C.

*Scanning Electron Microscopy and X-Ray Microanalysis*, 4th edition, Springer, 2018. DOI: [10.1007/978-1-4939-6676-9](https://doi.org/10.1007/978-1-4939-6676-9).

Supports:

- treating charging and image defects as acquisition artifacts worth stress-testing;
- the decision to expose the charging-gradient and edge-effect terms as engineering approximations;
- the limitation that the synthetic model is not a substitute for real tool data.

## [3] Lewis, J. P.

J. P. Lewis, “Fast Normalized Cross-Correlation,” *Vision Interface*, 1995, Industrial Light & Magic. Public technical report: [ncorr.com/download/publications/lewisfast.pdf](https://www.ncorr.com/download/publications/lewisfast.pdf).

Supports:

- normalized cross-correlation for template matching under local gain and bias changes;
- integral-image window statistics paired with transform-domain correlation;
- the FFT-ZNCC baseline used in `src/drift_sense/matching.py`.

## [4] Mallat, S.

Stéphane Mallat, *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd edition, Academic Press, 2008. Publisher page: [Elsevier](https://www.sciencedirect.com/book/9780123743701/a-wavelet-tour-of-signal-processing).

Supports:

- the general engineering practice of separating coarse-scale search from fine localization;
- documenting the coarse-to-fine design as a computational choice, not a claim of semiconductor-specific validation.

