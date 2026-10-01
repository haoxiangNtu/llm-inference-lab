#!/bin/bash
# Enter an extracted docker image rootfs with GPU, PFS and caches bound in.
# Usage: [ROOTFS=/root/xxx_rootfs] [COMPAT=host|image|none] glm53_chroot.sh <command...>   (default: bash)
#   COMPAT=none  : use the host's 535 driver libs (fine for CUDA 12.x images)
#   COMPAT=host  : use /root/cuda-compat-13/... from the host (needed for CUDA 13 images on driver 535)
#   COMPAT=image : use the image's own /usr/local/cuda*/compat
set -e
R=${ROOTFS:-/root/vllm_glm53_rootfs}
PFS=/mnt/pfs/4n3evq
CACHE=${CACHE:-/root/glm53_cache}
HOST_COMPAT=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat
mkdir -p $CACHE/vllm $CACHE/triton $CACHE/hf $R/glm53_cache $R$PFS $R/usr/local/nvidia/lib64 $R/proc $R/sys $R/dev $R/tmp $R/opt/host-compat

mnt() { mountpoint -q "$1" || mount "${@:2}" "$1"; }
mnt $R/proc -t proc proc
mnt $R/sys -t sysfs sys
mnt $R/dev --rbind /dev
mnt $R$PFS --bind $PFS
mnt $R/glm53_cache --bind $CACHE
mnt $R/tmp -t tmpfs -o size=64g tmpfs
[ -d $HOST_COMPAT ] && mnt $R/opt/host-compat --bind $HOST_COMPAT

# host NVIDIA driver user-space libraries (what nvidia-container-toolkit would inject)
if [ ! -e $R/usr/local/nvidia/lib64/libcuda.so.1 ]; then
  for lib in $(ldconfig -p | awk '/libcuda\.so|libnvidia-ml\.so|libnvidia-ptxjitcompiler|libnvidia-nvvm|libnvidia-fatbinaryloader|libcudadebugger|libnvidia-cfg|libnvidia-gpucomp|libnvidia-pkcs11/ {print $NF}' | sort -u); do
    cp -L "$lib" $R/usr/local/nvidia/lib64/ 2>/dev/null || true
  done
  for b in nvidia-smi; do p=$(command -v $b || true); [ -n "$p" ] && cp -L "$p" $R/usr/local/nvidia/ 2>/dev/null || true; done
fi
cp -L /etc/resolv.conf $R/etc/resolv.conf 2>/dev/null || true
cp -L /etc/hosts $R/etc/hosts 2>/dev/null || true
# pip mirror inside the image
mkdir -p $R/root/.config/pip $R/root/.pip
printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple\n[install]\ntrusted-host = mirrors.aliyun.com\n' > $R/root/.pip/pip.conf
cp $R/root/.pip/pip.conf $R/root/.config/pip/pip.conf

# image ENV -> array (values may contain spaces)
ENVFILE=$(mktemp)
python3 - "$R/.image_config.json" > "$ENVFILE" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1]))["config"]["Env"]:
    k = e.split("=", 1)[0]
    if k in ("PATH", "LD_LIBRARY_PATH"): continue
    print(e)
PY
envargs=()
while IFS= read -r line; do envargs+=("$line"); done < "$ENVFILE"
rm -f "$ENVFILE"

case "${COMPAT:-none}" in
  host)  LDP=/opt/host-compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib64 ;;
  image) IC=$(ls -d $R/usr/local/cuda-*/compat 2>/dev/null | head -1); LDP=${IC#$R}:/usr/local/cuda/lib64:/usr/local/nvidia/lib64 ;;
  *)     LDP=/usr/local/nvidia/lib64:/usr/local/cuda/lib64 ;;
esac
[ $# -eq 0 ] && set -- bash
exec chroot $R /usr/bin/env -i HOME=/root TERM=xterm "${envargs[@]}" \
  PATH=/usr/local/nvidia:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=$LDP \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/glm53_cache/hf XDG_CACHE_HOME=/glm53_cache \
  TRITON_CACHE_DIR=/glm53_cache/triton VLLM_CACHE_ROOT=/glm53_cache/vllm PYTHONUNBUFFERED=1 \
  "$@"
