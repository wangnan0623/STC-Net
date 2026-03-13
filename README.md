# Enhancing Object Tracking via Spatio-Temporal Collaboration in Frame-Event Networks
Code for "**Enhancing Object Tracking via Spatio-Temporal Collaboration in Frame-Event Networks**"  (_**The Visual Computer**_). 

Nan Wang, Yong Song*, Li Wang, Yingbo He, Shaoxing Wu, Ya Zhou, Teng Luo, Shuang Wang, Linjun Zeng

School of Optics and Photonics, Beijing Institute of Technology, Beijing

Beijing Institute of Control Engineering, Beijing
## Abstract
Object tracking in challenging conditions, such as high-speed motion and extreme lighting, demands robust perception capabilities. Event cameras, with their high temporal resolution and dynamic range, complement conventional frame-based sensors. Leveraging both frame and event modalities, we propose the Spatio-Temporal Collaboration Network (STC-Net) to deeply integrate spatial and temporal information. Our approach introduces an Adaptive Spiking Neuron (ASN) module to process raw event streams, preserving high temporal resolution while filtering noise. A High-order Spatio-Temporal Fusion (HSTF) module aligns and fuses RGB semantics with event dynamics. Experiments on the FE108 and VisEvent datasets demonstrate state-of-the-art performance, outperforming existing RGB-Event trackers by 3.7\% in RPR and 2.0\% in RSR on the FE108 benchmark. Ablation studies validate the effectiveness of each key component.
<img width="6012" height="2260" alt="f1" src="https://github.com/user-attachments/assets/9084c0fb-e9f8-43b0-bf7f-088feffa45ae" />
For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact 3120240652@bit.edu.cn and yongsong@bit.edu.cn.
## Installation
This code is based on Python 3.8 and PyTorch 2.0

1. We recommend using conda to build the environment:
  ```
  conda create -n STC_Net python=3.8
  
  conda activate STC_Net
  
  conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
2. Install the dependent packages:
  ```
  pip install -r requirements.txt
  ```
3. Install deformable convolution according to [EDVR](https://github.com/xinntao/EDVR):
    ```
    python setup.py develop
    ```
