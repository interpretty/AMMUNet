# AMMUNet: Multi-Scale Attention Map Merging for Remote Sensing Image Segmentation
## Paper Link

https://arxiv.org/abs/2404.13408

## Code Usage Guide

This project is based on MMSegmentation.

Currently, the baseline configuration has been uploaded. The code for AMMUNet and the improved version based on UNetFormer will be uploaded soon.

1. Please follow the installation instructions in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and configure the relevant virtual environment.

2. Copy the 'AMMUNet/mmsegmentation' folder from this project to 'local-path/mmsegmentation'.

For the file 'mmsegmentation/mmseg/models/decode_heads/init.py', manually modify it by adding the following code snippet at the end of the file 'local-path/mmsegmentation/mmseg/models/decode_heads/init.py':

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
