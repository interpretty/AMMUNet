# AMMUNet: Multi-Scale Attention Map Merging for Remote Sensing Image Segmentation

## Paper Information
- **Article title**: AMMUNet: Multi-Scale Attention Map Merging for Remote Sensing Image Segmentation
- **Journal acronym**: LGRS
- **Article DOI**: 10.1109/LGRS.2024.3506718
- **Paper link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10767738)
- **arXiv link**: [arXiv:2404.13408](https://arxiv.org/abs/2404.13408)

## Code Usage Guide
This project is based on MMSegmentation.

The full implementation of AMMUNet and the improved version based on UNetFormer have been uploaded.

1. Please follow the installation instructions in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and configure the relevant virtual environment.

2. Copy the 'AMMUNet/mmsegmentation' folder from this project to 'local-path/mmsegmentation'.

3. For the file 'mmsegmentation/mmseg/models/decode_heads/init.py', manually modify it by adding the following code snippet at the end of the file 'local-path/mmsegmentation/mmseg/models/decode_heads/init.py':

```python
from .ammunet_head import AMMUNetHead
from .ammunetformer_head import AMMUNetFormerHead
from .msaunet_head import MSAUNetHead
from .pfmsaunet_head import PfMSAUNetHead
from .pfmsaunetformer_head import PfMSAUNetFormerHead
from .unet_head import UNetHead
from .unetformer_head import UNetFormerHead

__all__ = __all__ + ['AMMUNetHead', 'AMMUNetFormerHead', 'MSAUNetHead', 'PfMSAUNetHead', 'PfMSAUNetFormerHead',
                     'UNetHead', 'UNetFormerHead']
```

If you encounter any issues, please raise them in the Issues section.
