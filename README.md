This page is the code of our paper (in submission) "Intra-and-inter Instance Location Correlation Network for Human Object Interaction Detection".
## preparation
the dependency below are all required for this project  
CUDA>=9.2, GCC>=5.4,Python>=3.7  
you also need to create a new conda environment  
`conda create -n ILCN python=3.8  
conda activate ILCN  
pip install -r requirements.txt  
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch` 
then, clone this project:  
`git clone https://github.com/lumao23/ILCN.git`  
`cd ILCN`    
Then, compile Multi-Scale Deformable Attention, [MSDA](https://github.com/fundamentalvision/Deformable-DETR)  
`cd ./models/ops`  
`sh ./make.sh`

