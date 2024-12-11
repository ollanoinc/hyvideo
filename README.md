```
 _                 _     _
| |__  _   ___   _(_) __| | ___  ___
| '_ \| | | \ \ / / |/ _` |/ _ \/ _ \
| | | | |_| |\ V /| | (_| |  __/ (_) |
|_| |_|\__, | \_/ |_|\__,_|\___|\___/
       |___/
```

# 👋 HyVideo

This project is a first step in integrating [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) into [Diffusers](https://github.com/huggingface/diffusers).

The License is inherted from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).


## Installation

```bash
pip install hyvideo
```

## Usage

Please note that you need at least 80GB VRAM to run this pipeline. CPU offloading is having issues at the moment (PRs welcome!).

```python
import torch
from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline
from hyvideo.modules.models import HYVideoDiffusionTransformer
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D

pipe = HunyuanVideoPipeline.from_pretrained(
    'magespace/hyvideo-diffusers',
    transformer=HYVideoDiffusionTransformer.from_pretrained(
        'magespace/hyvideo-diffusers',
        torch_dtype=torch.bfloat16,
        subfolder='transformer'
    ),
    vae=AutoencoderKLCausal3D.from_pretrained(
        'magespace/hyvideo-diffusers',
        torch_dtype=torch.bfloat16,
        subfolder='vae'
    ),
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to('cuda')
pipe.vae.enable_tiling()
```

Then running:

```python
prompt = "Close-up, A little girl wearing a red hoodie in winter strikes a match. The sky is dark, there is a layer of snow on the ground, and it is still snowing lightly. The flame of the match flickers, illuminating the girl's face intermittently."

result = pipe(prompt)
```

Post-processing:

```python
import PIL.Image
from diffusers.utils import export_to_video
output = result.videos[0].permute(1, 2, 3, 0).detach().cpu().numpy()
output = (output * 255).clip(0, 255).astype("uint8")
output = [PIL.Image.fromarray(x) for x in output]

export_to_video(output, "output.mp4", fps=24)
```
