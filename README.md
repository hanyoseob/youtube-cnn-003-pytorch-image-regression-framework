# Image regression framework 구축하기 (009 ~ 010)
[![초보 딥러닝 강의-009 image regression framework 구축하기](https://i.ytimg.com/vi/qn3gc-gQDFQ/sddefault.jpg)](https://www.youtube.com/watch?v=qn3gc-gQDFQ)

[![초보 딥러닝 강의-010 학습하면 뭐라도 된다](https://i.ytimg.com/vi/XNE5Up5pCgE/sddefault.jpg)](https://www.youtube.com/watch?v=XNE5Up5pCgE)


## Denoising

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task denoising \
            --opts random 30.0


## Inpainting

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task inpainting \
            --opts uniform 0.5
---

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task inpainting \
            --opts random 0.5


## Super resolution

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task super_resolution \
            --opts bilinear 4.0
