
SimSwap Windows (Anaconda + Python 3.10) — Self-Host Quickstart
================================================================

This guide captures the working setup we debugged together, tailored for Windows with Anaconda (Python 3.10), NumPy 1.26.x, MoviePy 2.x, modern InsightFace (buffalo_l), and optional GPU acceleration. It includes one-time patches and the exact commands to run the built-in demo and your own inputs.

What we fixed / decided (TL;DR)
-------------------------------
- Pin **NumPy to 1.26.4** (NumPy 2.x breaks older wheels like OpenCV/ONNX Runtime).
- Use **MoviePy 2.x** (no `moviepy.editor` submodule): replace `.set_audio(...)` → `.with_audio(...)`.  
- Use **InsightFace 0.7.3** with `FaceAnalysis('buffalo_l')` (instead of legacy ONNX packs like `antelope`).
- Use **PyTorch cu118** wheels and **onnxruntime-gpu==1.16.3** for CUDA acceleration (optional but recommended).
- (Optional) `--use_mask` uses a face-parsing checkpoint to blend faces more naturally.

Prerequisites
-------------
- **Anaconda** installed.
- (Optional GPU) Latest **NVIDIA driver** (verify with `nvidia-smi`).
- **git** and **curl** in PATH (or use your browser to download model files).
- Use **Anaconda Prompt (cmd)** for all commands below.

1) Create environment & get the code
------------------------------------
```
conda create -n simswap310 python=3.10 -y
conda activate simswap310

git clone https://github.com/neuralchen/SimSwap
cd SimSwap
```

2) Pin NumPy < 2 for this project
---------------------------------
```
python -c "open('constraints.txt','w',encoding='utf-8').write('numpy==1.26.4\n')"
set PIP_CONSTRAINT=%CD%\constraints.txt
```

3) Install core libraries (compatible with NumPy 1.26.4)
--------------------------------------------------------
```
pip install --no-cache-dir -c constraints.txt ^
  pillow==10.3.0 imageio==2.37.0 imageio-ffmpeg==0.4.9 ^
  opencv-python-headless==4.9.0.80 scikit-image==0.25.2 onnx==1.19.0 ^
  "moviepy>=2.1,<2.3"
```

4) PyTorch (CUDA 11.8 build) — GPU recommended
----------------------------------------------
GPU build (preferred):
```
pip install --index-url https://download.pytorch.org/whl/cu118 -c constraints.txt ^
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```
CPU-only (if you don’t have NVIDIA GPU):
```
pip install --index-url https://download.pytorch.org/whl/cpu -c constraints.txt ^
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

5) ONNX Runtime + InsightFace
------------------------------
GPU (preferred):
```
pip install --no-cache-dir -c constraints.txt onnxruntime-gpu==1.16.3 insightface==0.7.3
```
CPU:
```
pip install --no-cache-dir -c constraints.txt onnxruntime==1.16.3 insightface==0.7.3
```

6) Download model files
-----------------------
ArcFace (required):
```
mkdir arcface_model 2>nul
curl -L -o arcface_model\arcface_checkpoint.tar ^
  https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
```
Face parsing (optional; needed for `--use_mask`):
```
mkdir parsing_model\checkpoint 2>nul
curl -L -o parsing_model\checkpoint\79999_iter.pth ^
  https://github.com/neuralchen/SimSwap/releases/download/1.0/79999_iter.pth
```

7) Patch SimSwap for MoviePy 2.x + modern InsightFace
-----------------------------------------------------
A) MoviePy 2.x: replace `.set_audio(` → `.with_audio(` in the code path:
```
python -c "import io; p=r'util\\videoswap.py'; s=io.open(p,'r',encoding='utf-8').read(); s=s.replace('.set_audio(','.with_audio('); io.open(p,'w',encoding='utf-8').write(s); print('patched',p)"
```

B) Replace `insightface_func\face_detect_crop_single.py` with a FaceAnalysis-based version that works with InsightFace 0.7.x (buffalo_l):
```
python -c "open(r'insightface_func\\face_detect_crop_single.py','w',encoding='utf-8').write(r'''# -*- coding: utf-8 -*-
from __future__ import division
import collections, glob, os, os.path as osp
import numpy as np, cv2
from insightface.app import FaceAnalysis
from insightface_func.utils import face_align_ffhqandnewarc as face_align

__all__ = ['Face_detect_crop','Face']
Face = collections.namedtuple('Face',['bbox','kps','det_score','embedding','gender','age','embedding_norm','normed_embedding','landmark'])
Face.__new__.__defaults__ = (None,)*len(Face._fields)

class Face_detect_crop:
    def __init__(self, name='buffalo_l', root='~/.insightface'):
        self.models = {}
        self.mode = 'None'
        cache_root = os.path.expanduser(root) if root else os.path.expanduser('~/.insightface')
        pack = 'buffalo_l' if not name else {'antelope':'buffalo_l','antelopev2':'buffalo_l'}.get(name,name)
        print(f\"No local ONNX used; using FaceAnalysis pack '{pack}' with cache '{cache_root}'\")
        self.app = FaceAnalysis(name=pack, root=cache_root, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640,640))
        self.models = self.app.models
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640,640), mode='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        print('set det-size:', det_size)
        self.det_size = det_size
        for task, model in self.models.items():
            if task=='detection':
                try:
                    model.prepare(ctx_id, det_thresh=det_thresh, input_size=det_size)
                except TypeError:
                    model.prepare(ctx_id, input_size=det_size)
                    if hasattr(model,'threshold'): model.threshold = det_thresh
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        try:
            bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        except TypeError:
            try: bboxes, kpss = self.det_model.detect(img, max_num=max_num)
            except TypeError: bboxes, kpss = self.det_model.detect(img)
        if bboxes is None or (hasattr(bboxes,'shape') and bboxes.shape[0]==0):
            return None
        best_index = int(np.argmax(bboxes[...,4])) if hasattr(bboxes,'ndim') and bboxes.ndim==2 and bboxes.shape[1]>=5 else 0
        kps = kpss[best_index] if kpss is not None else None
        M,_ = face_align.estimate_norm(kps, crop_size, mode=self.mode)
        align_img = cv2.warpAffine(img, M, (crop_size,crop_size), borderValue=0.0)
        return [align_img],[M]
''') and print('wrote Face_detect_crop')"
```

8) Sanity checks
----------------
```
python -c "import numpy, torch, onnxruntime as ort, insightface; \
print('numpy', numpy.__version__); \
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); \
print('ORT providers', ort.get_available_providers()); \
print('insightface', insightface.__version__)"
```
- Expect: `numpy 1.26.4`, `cuda? True` (if GPU), ORT providers include `CUDAExecutionProvider`, and `insightface 0.7.3`.

9) Run the built-in demo
------------------------
```
python test_video_swapsingle.py ^
  --crop_size 224 ^
  --use_mask ^
  --name people ^
  --Arc_path arcface_model\arcface_checkpoint.tar ^
  --pic_a_path .\demo_file\Iron_man.jpg ^
  --video_path .\demo_file\multi_people_1080p.mp4 ^
  --output_path .\output\multi_test_swapsingle.mp4 ^
  --temp_path .\temp_results
```
- If you have the 512 weights at `.\checkpoints\simswap\simswap_512.pt`, switch `--crop_size 512`.

10) Use your own image/video
----------------------------
```
set "SRC=C:\path\to\your_face.jpg"
set "VID=C:\path\to\your_video.mp4"
set "OUT=output\your_video_swapped.mp4"
python test_video_swapsingle.py ^
  --crop_size 224 ^
  --use_mask ^
  --name people ^
  --Arc_path arcface_model\arcface_checkpoint.tar ^
  --pic_a_path "%SRC%" ^
  --video_path "%VID%" ^
  --output_path "%OUT%" ^
  --temp_path .\temp_results
```

11) Optional quality tweaks
---------------------------
- **512 crop** (requires `checkpoints\simswap\simswap_512.pt`) → better detail, higher VRAM.
- **Detector input 1024** for crisper landmarks (slower):
  ```
  python -c "import io,re; p=r'insightface_func\\face_detect_crop_single.py'; s=io.open(p,'r',encoding='utf-8').read(); s=re.sub(r'det_size=\(640,\\s*640\)','det_size=(1024,1024)',s); io.open(p,'w',encoding='utf-8').write(s); print('det_size set to 1024x1024')"
  ```
- **Keep original resolution & remove watermark** (optional):
  - Remove all watermark uses (safe comment-out):
    ```
    python - << "PY"
    import io, re
    p=r"util\\reverse2original.py"
    s=io.open(p,"r",encoding="utf-8").read().splitlines()
    out=[]
    for line in s:
        if ("logoclass" in line) and ("def " not in line) and ("class " not in line) and ("import " not in line):
            out.append("# [no-watermark] " + line)
        else:
            out.append(line)
    io.open(p,"w",encoding="utf-8").write("\n".join(out))
    print("watermark calls disabled in", p)
    p=r"util\\videoswap.py"
    s=io.open(p,"r",encoding="utf-8").read()
    s=re.sub(r",\\s*logoclass\\s*,", ", None,", s)
    s=re.sub(r"logoclass\\s*=", "logoclass_IGNORED=", s)
    io.open(p,"w",encoding="utf-8").write(s)
    print("forced logoclass=None at call sites in", p)
    PY
    ```
  - Ensure high-quality encode and original FPS (tune `write_videofile`):
    ```
    python - << "PY"
    import io,re
    p=r'util\\videoswap.py'
    s=io.open(p,'r',encoding='utf-8').read()
    s=re.sub(r"write_videofile\\(\\s*output_path\\s*(?:,[^)]+)?\\)",
             "write_videofile(output_path, fps=int(video_audio_clip.fps), codec='libx264', audio_codec='aac', bitrate='12M', preset='slow')",
             s, count=1)
    io.open(p,'w',encoding='utf-8').write(s)
    print('write_videofile tuned in',p)
    PY
    ```

Troubleshooting tips
--------------------
- Always install new packages with the constraint so NumPy stays < 2:
  ```
  pip install -c constraints.txt <packages...>
  ```
- Version check one-liner:
  ```
  python -c "import numpy,torch,onnxruntime as ort,moviepy,PIL,cv2,insightface; \
  print('numpy',numpy.__version__); print('torch',torch.__version__,'cuda?',__import__('torch').cuda.is_available()); \
  print('ORT providers',ort.get_available_providers()); print('moviepy',moviepy.__version__); \
  print('Pillow',PIL.__version__); print('cv2',cv2.__version__); print('insightface',insightface.__version__)"
  ```
- If ONNX Runtime shows only CPU provider on a GPU machine:
  - Ensure `onnxruntime-gpu==1.16.3` is installed in this env.
  - Verify `nvidia-smi` works; install CUDA 11.8 runtime/required DLLs if needed.
- If MoviePy errors (e.g., `.set_audio` missing), replace with `.with_audio`.
- If you see `np.float` / `np.int` errors, replace with built-ins (`float`, `int`) or specific dtypes (e.g., `np.float32`).

Happy swapping!
