# Usage: source /root/go2-supermimic/activate-cpu.sh
source /root/.venvs/go2-cpu/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export MUJOCO_GL=egl
cd /root/go2-supermimic
