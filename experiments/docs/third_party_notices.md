# Third-party notices

## Restormer

The `src/ihc_restormer.py` architecture is adapted from the official
[Restormer](https://github.com/swz30/Restormer) implementation.

MIT License

Copyright (c) 2022 Syed Waqas Zamir and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DINOv2

Round112 uses the official Meta DINOv2 implementation at revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and the ViT-S/14 register model
pretrained on LVD142M. Code and DINOv2 weights are Apache-2.0 licensed.
Source: https://github.com/facebookresearch/dinov2
The pinned source is prepared under `transfers/dinov2` with its upstream
LICENSE intact; `tools/prepare_dinov2.py` downloads and validates the weights.
Cell-DINO and XRay-DINO weights are not used by this model.
