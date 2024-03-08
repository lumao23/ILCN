This page is the code of our paper (in submission) "Intra-and-inter Instance Location Correlation Network for Human Object Interaction Detection".
## preparation
the dependency below are all required for this project
Linux, CUDA>=9.2, GCC>=5.4  
Python>=3.7,  
`pip install -r requirements.txt`  
`pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113` or  
`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`  
then, clone this project:  
`git clone https://github.com/lumao23/ILCN.git`  
`cd ILCN`  
Then, compile Multi-Scale Deformable Attention, [MSDA](https://github.com/fundamentalvision/Deformable-DETR)  
`cd ./models/ops`  
`sh ./make.sh`

