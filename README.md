# 👋 HyVideo

This project is a first step in integrating [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) into [Diffusers](https://github.com/huggingface/diffusers).

**All credit go to [Tencent](https://github.com/Tencent) for the original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) project.**

**Thank you to Huggingface for the [Diffusers](https://github.com/huggingface/diffusers) library.** Special shout-out to [@a-r-r-o-w](https://github.com/a-r-r-o-w) for his work on integrating HunyuanVideo.

The License is inherted from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).

This library is provided as-is and will be superseded by the official release of HunyuanVideo via [Diffusers](https://github.com/huggingface/diffusers). Please help out if you can on the [PR](https://github.com/huggingface/diffusers/pull/10136).


## Installation

```bash
pip install git+https://github.com/ollanoinc/hyvideo.git
```

You will also need to install [flash-attn](https://github.com/Dao-AILab/flash-attention) for now.

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

For faster generation, you can optimize the `transformer` with `torch.compile`. Additionally, increasing `shift` in the scheduler can allow for lower step values as shown in the original paper.

Generation time is quadratic with the number of pixels, so reducing the height and width and decreasing the number of frames will drastically speed up generation at the price of video quality.
