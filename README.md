# A benchmark for sparse 3D conv python libraries


- install sptr

```
conda create -n t3.8 python=3.8;
conda activate t3.8;
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter==2.0.9
pip install torch_geometric==1.7.2
pip install torch_cluster
pip install torch_sparse
pip install timm
git clone https://github.com/dvlab-research/SparseTransformer.git
cd SparseTransformer
python setup.py install
```




- install mmloc

```
conda create -n t3.8 python=3.8;
conda activate t3.8;
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install scikit-learn
pip install tqdm
pip install pytorch-metric-learning==1.1
pip install tensorboard
```




- Minkowski Engine
  https://github.com/NVIDIA/MinkowskiEngine

  ```
  conda create -n py3-mink python=3.8
  conda activate py3-mink
  
  conda install openblas-devel -c anaconda
  conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
  
  # Install MinkowskiEngine
  
  # Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
  # export CUDA_HOME=/usr/local/cuda-11.1
  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
  
  # Or if you want local MinkowskiEngine
  git clone https://github.com/NVIDIA/MinkowskiEngine.git
  cd MinkowskiEngine
  python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
  ```

  

- SpConv
  https://github.com/traveller59/spconv

  ```
  pip install spconv-cu118	
  ```

  

- TorchSparse
  https://github.com/mit-han-lab/torchsparse

  ```
  sudo apt-get install libsparsehash-dev
  ```

  ```
  pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
  ```

  
