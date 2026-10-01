#!/bin/bash
# Run INSIDE the vllm-backport chroot. Mirrors docker/Dockerfile.sm80 of Mrzhiyao/glm53-a800-vllm:
# build the SM80-only vLLM CUDA extension from the patched source and install the wheel.
# All CMake external projects come from pre-cloned local checkouts (GitHub is unreachable from the pod).
set -euo pipefail
W=/mnt/pfs/4n3evq/lhx/llm_deploy
SRC=${SRC:-$W/vllm-src}
D=$W/deps
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=${MAX_JOBS:-48} NVCC_THREADS=${NVCC_THREADS:-2} \
       SETUPTOOLS_SCM_PRETEND_VERSION=0.8.1+glm53a800 CCACHE_DISABLE=1 GIT_TERMINAL_PROMPT=0
export VLLM_CUTLASS_SRC_DIR=$D/cutlass \
       VLLM_FLASH_ATTN_SRC_DIR=$D/flash-attention \
       FLASH_MLA_SRC_DIR=$D/FlashMLA \
       FLASH_KDA_SRC_DIR=$D/FlashKDA \
       FMHA_SM100_SRC_DIR=$D/MSA \
       TML_FA4_SRC_DIR=$D/tml-fa4 \
       DEEPGEMM_SRC_DIR=$D/DeepGEMM \
       QUTLASS_SRC_DIR=$D/qutlass \
       TRITON_KERNELS_SRC_DIR=$D/triton/python/triton_kernels/triton_kernels
for d in $VLLM_CUTLASS_SRC_DIR $VLLM_FLASH_ATTN_SRC_DIR $FLASH_MLA_SRC_DIR $FLASH_KDA_SRC_DIR $FMHA_SM100_SRC_DIR $TML_FA4_SRC_DIR $DEEPGEMM_SRC_DIR $QUTLASS_SRC_DIR $TRITON_KERNELS_SRC_DIR; do
  test -d "$d" || { echo "missing dependency checkout: $d"; exit 1; }
done
echo "== base image vllm: $(python3 -c 'import vllm;print(vllm.__version__)')  nvcc: $(nvcc --version | tail -1)"
python3 -m pip install -q --root-user-action=ignore 'cmake>=3.26.1' 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' wheel ninja
NVRTC=$(ls /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.1[0-9] | head -1)
ln -sf "$NVRTC" /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
BUILD=/root/vllm-build
rm -rf $BUILD && mkdir -p $BUILD && cp -a $SRC/. $BUILD/ && cd $BUILD
DIST=/usr/local/lib/python3.12/dist-packages/vllm
cp -a $DIST/vllm-rs vllm/vllm-rs && cp -a $DIST/_rust_*.so vllm/
mkdir -p /tmp/wheels
export VERBOSE=1
time python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/wheels . 2>&1 | stdbuf -oL grep -vE "^\s*(copying|creating) "
ls -la /tmp/wheels
python3 -m pip install --no-deps --force-reinstall --root-user-action=ignore /tmp/wheels/vllm-*.whl
cp /tmp/wheels/vllm-*.whl $W/ 2>/dev/null || true
python3 -c "import vllm; print('installed vllm', vllm.__version__)"
echo "BUILD_DONE"
