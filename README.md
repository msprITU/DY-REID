# DY-REID
In this repo, you can find the [**model files**](https://drive.google.com/drive/folders/16O3ncmayQI6HPaX32zfn13lz6_t_hme6?usp=sharing) created by training with DY-Cace and DY-BL on four different datasets. 
The ImageNet pre-trained model for dynamic ResNet-50 can be downloaded [**here**](https://drive.google.com/file/d/14VUXecopj3aTu1s4IKdT2FsPt7Iq7BDK/view). Model needs to be placed in _pretrained_models_ directory. 
Also Training/Inference code and configuration files will be added.

# Person Re-identification Datasets: 

- [**Market-1501**](https://www.v7labs.com/open-datasets/market-1501) 
- [**CUHK03**](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- [**DukeMTMC-reID**](https://exposing.ai/duke_mtmc/)


# Reference papers and repositories : 

Baseline and CaceNet person ReID models: https://github.com/TencentYoutuResearch/PersonReID-YouReID

Yu, F., Jiang, X., Gong, Y., Zhao, S., Guo, X., Zheng, W.-S., Zheng,  F., Sun, X.: Devil’s in the details: Aligning visual clues for conditional embedding in person re-identification. In: Proc. IEEE CVPR (2021)

Dynamic convolution : https://github.com/liyunsheng13/dcd 

Chen, Y., Dai, X., Liu, M., Chen, D., Yuan, L., Liu, Z.: Dynamic convolution: Attention over convolution kernels. In: Proc. IEEE CVPR, pp.11030–11039 (2020)
