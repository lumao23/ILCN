# ILCN
This page is the code of our paper (in submission) "Intra-and-inter Instance Location Correlation Network for Human Object Interaction Detection".


## Installation
Our model is run under ```CUDA>=9.2, GCC>=5.4,Python>=3.7  ```. Other versions might be available as well.

1. Clone this repo
```sh
git clone https://github.com/lumao23/ILCN.git
cd ILCN
```

2. Install Pytorch and torchvision
```sh
conda create -n ILCN python=3.8
conda activate ILCN
pip install -r requirements.txt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Or you can follow the instruction on https://pytorch.org/get-started/locally/ for another pytorch version

2. Compiling CUDA operators
```sh
cd models/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../..
```

## Data


## Run



### To support fp16
* models/ops/modules/ms_deform_attn.py
* models/ops/functions/ms_deform_attn_func.py

### To fix a pytorch version bug
* util/misc.py
