# Path to the downloaded CLIP official weights.
# See: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30
# CLIP_VIT_B16_PATH = '/work/piyush/pretrained_checkpoints/CLIP/weights/clip_vit_b_16.pth'
# CLIP_VIT_L14_PATH = '/work/piyush/pretrained_checkpoints/CLIP/weights/clip_vit_l_32.pth'
CLIP_VIT_B16_PATH = '/work/piyush/pretrained_checkpoints/OpenAI-CLIP/ViT-B-16.pt'
CLIP_VIT_L14_PATH = '/work/piyush/pretrained_checkpoints/OpenAI-CLIP/ViT-L-14.pt'

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True

# Configuration of datasets. The required fields are listed for Something-something-v2 (ssv2)
# and Kinetics-400 (k400). Fill in the values to use the datasets, or add new datasets following
# these examples.
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='',
        VAL_LIST='',
        NUM_CLASSES=174,
    ),
    'k400': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='',
        VAL_LIST='',
        NUM_CLASSES=400,
    ),
}
