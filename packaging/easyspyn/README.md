# EasySpyn (EasySpyn): EasySpin's EPR simulators in Python

A Python port of [EasySpin](https://easyspin.org), the MATLAB toolbox for EPR spectral simulation by Stefan Stoll
and co-workers (MIT licence), keeping EasySpin's interface: spin systems as dictionaries with EasySpin's field
names, fields in mT, couplings in MHz, 1-based spin numbers.

    pip install easyspyn
    pip install "easyspyn[fast]"  # adds Numba for the line-summation and projection kernels

Nothing to install? Run EasySpyn in the browser (https://ajerschow.github.io/spynach/try.html), in Colab
(https://colab.research.google.com/github/ajerschow/spynach/blob/main/docs/colab/easyspyn-quick-start.ipynb) or on Binder
(https://mybinder.org/v2/gh/ajerschow/spynach/main?urlpath=lab/tree/docs/notebooks/).

```python
import easyspyn as es

nitroxide = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "lwpp": 0.3}
powder = es.pepper(nitroxide, {"mwFreq": 9.5, "Range": [330, 350], "nPoints": 1024})
print(powder.x.shape, powder.spec.shape)
```

What is ported: `pepper` and `garlic` (cw EPR of powders, crystals and solutions), `esfit` (fitting), `salt` and
`endorfrq` (ENDOR), `curry` (magnetometry), `chili` (slow motion), `saffron` (pulse EPR: ESEEM, HYSCORE, Mims
ENDOR), and the foundation underneath (`sop`, `ham`, `levels`, `resfields`, line shapes, grids). Results are
compared with EasySpin 6.0.15's own output on about 280 reference cases. Not ported: `saffron`'s `thyme` engine
and `spidyan`, the GUIs, and some options; the guide lists them.

EasySpyn is part of the [Spynach](https://github.com/ajerschow/spynach) project, which also ports the Spinach
spin dynamics kernel. It is an early release (0.3.0); corrections and bug reports are welcome on GitHub.

- Guides: https://github.com/ajerschow/spynach/blob/main/docs/easyspyn.md
- Differences from EasySpin: https://github.com/ajerschow/spynach/blob/main/docs/coming-from-easyspin.md
- Try it in the browser, with timings against EasySpin: https://ajerschow.github.io/spynach/easyspin.html

Attribution: EasySpyn is a translation of EasySpin and keeps its notice; see `LICENSE`.
Each function's docstring names the EasySpin contributors to the original file (`help(easyspyn.pepper)`); the full table is at https://github.com/ajerschow/spynach/blob/main/docs/easyspin-credits.md.
