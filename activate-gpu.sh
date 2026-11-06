source /root/.venvs/go2-gpu/bin/activate
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
export MUJOCO_GL=egl
cd /root/go2-supermimic

export LD_LIBRARY_PATH=/root/.venvs/go2-gpu/lib/python3.13/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
