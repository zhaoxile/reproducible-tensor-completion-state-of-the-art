# reproducible-tensor-completion-state-of-the-art
Collection of popular and reproducible tensor completion works.

Criteria: works must have codes available, and the reproducible results demonstrate state-of-the-art performances.


**This list is maintained by:**

**[Xi-Le Zhao](https://zhaoxile.github.io/),  [Tai-Xiang Jiang](https://taixiangjiang.github.io/),  [Yu-Bang Zheng](https://yubangzheng.github.io/)** **[[Email]](https://zhaoxile.github.io/) (UESTC)**


## Excellent Review
* Tensor Decompositions and Applications (SIAM Review 2009), Tamara G. Kolda and Brett W. Bader  
* Tensor Decompositions for Signal Processing Applications: From Two-way to Multiway Component Analysis (IEEE SP 2015), Andrzej Cichocki et al.
* Tensor Decomposition for Signal Processing and Machine Learning (IEEE TSP 2017), Nicholas D. Sidiropoulos et al.
* Sparse Recovery: From Vectors to Tensors (National Science Review 2017), Yao Wang et al.
* Low Rank Tensor Completion for Multiway Visual Data (Signal Processing 2019), Zhen Long et al.
* Tensor Completion Algorithms in Big Data Analytics (TKDD 2019), Qingquan Song et al.

## CANDECOMP/PARAFAC Decomposition
 * BCPF [[Web]](https://qibinzhao.github.io/) [[Code]](https://github.com/qbzhao/BCPF) [[PDF]](https://arxiv.org/abs/1401.6497)
   * Bayesian CP factorization of incomplete tensors with automatic rank determination (PAMI 2015), Qibin Zhao et al.


## Tucker Decomposition
* SNN [[Web]](https://www.cs.rochester.edu/u/jliu/publications.html) [[Code]](http://peterwonka.net/Publications/code/LRTC_Package_Ji.zip) [[PDF]](https://www.cs.rochester.edu/u/jliu/paper/Ji-ICCV09.pdf)
   * Tensor Completion for Estimating Missing Values in Visual Data (PAMI 2012), Ji Liu et al.
* TMac [[Web]](https://www.math.ucla.edu/~wotaoyin/papers/tmac_tensor_recovery.html) [[Code]](https://www.math.ucla.edu/~wotaoyin/papers/tmac_tensor_recovery.html) [[PDF]](https://www.math.ucla.edu/~wotaoyin/papers/tmac_tensor_recovery.html)
  * Parallel matrix factorization for low-rank tensor completion (Inverse Problems and Imaging), Yangyang Xu et al.
* KBR [[Web]](http://gr.xjtu.edu.cn/web/dymeng/1;jsessionid=F03A6AE30867A1EE7DE9D577DD4E253D) [[Code]](https://github.com/XieQi2015/KBR-TC-and-RPCA) [[PDF]](https://ieeexplore.ieee.org/iel7/34/4359286/08000407.pdf)
   * Kronecker-Basis-Representation Based Tensor Sparsity and Its Applications to Tensor Recovery (PAMI 2017), Qi Xie et al.

  
## Tensor Singular Value Decomposition
* TRPCA [[Web]](https://canyilu.github.io/publications/) [[Code]](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) [[PDF]](https://arxiv.org/abs/1804.03728)
   * Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm (PAMI2018), Canyi Lu et al.
 * 3DTNN [[Web]](https://zhaoxile.github.io/) [[Code]](https://github.com/zhaoxile/Mixed-Noise-Removal-in-Hyperspectral-Image-via-Low-Fibered-Rank-Regularization) [[PDF]](https://zhaoxile.github.io/paper/2020/Mixed%20Noise%20Removal%20in%20Hyperspectral%20Image%20via%20Low-Fibered-Rank%20Regularization.pdf)
   * Mixed Noise Removal in Hyperspectral Image via Low-Fibered-Rank Regularization (IEEE TGRS2020), Yu-Bang Zheng et al.
* PSTNN [[Web]](https://sites.google.com/view/taixiangjiang/) [[Code]](https://github.com/zhaoxile/Multi-dimensional-imaging-data-recovery-via-minimizing-the-partial-sum-of-tubal-nuclear-norm) [[PDF]](https://zhaoxile.github.io/paper/2019/Multi-dimensional%20imaging%20data%20recovery%20via%20minimizing%20the%20partial%20sum%20of%20tubal%20nuclear%20norm.pdf)
   * Multi-dimensional imaging data recovery via minimizing the partial sum of tubal nuclear norm
 (JCAM2020), Tai-Xiang Jiang et al. 
* TNN [[Web]](http://www.ece.tufts.edu/~shuchin/) [[Code]](http://www.ece.tufts.edu/~shuchin/tensor_completion_and_rpca.zip) [[PDF]](https://www.zpascal.net/cvpr2014/Zhang_Novel_Methods_for_2014_CVPR_paper.pdf)
   * Novel Methods for Multilinear Data Completion and De-noising Based on Tensor-SVD (CVPR2014), Zemin Zhang et al.
   
## Tensor Network Decomposition (Maintained by Wen-Jie Zheng and Yu-Bang Zheng)
* TT Decomposition [[Web]](https://epubs.siam.org/doi/10.1137/090752286) [[Code]] [[PDF]](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition)
   * Tensor-Train Decomposition, _SIAM Journal on Scientific Computing 2011_, **I. V. Oseledets**
* TMac-TT [[Web]](https://sites.google.com/site/jbengua/home) [[Code]](https://sites.google.com/site/jbengua/home/projects/efficient-tensor-completion-for-color-image-and-video-recovery-low-rank-tensor-train/TMacTT_Images.zip?attredirects=0&d=1) [[PDF]](https://www.researchgate.net/publication/303821165_Efficient_Tensor_Completion_for_Color_Image_and_Video_Recovery_Low-Rank_Tensor_Train)
   * Efficient Tensor Completion for Color Image and Video Recovery: Low-Rank Tensor Train, _IEEE Transactions on Image Processing 2017_, **Johann A. Bengua et al.**
* MF-TTTV  [[Web]](https://mengding56.github.io/homepage/) [[Code]](https://mengding56.github.io/homepage/) [[PDF]](https://mengding56.github.io/homepage/)
   * Low-Rank Tensor Completion Using Matrix Factorization Based on Tensor Train Rank and Total Variation, _Journal of Scientific Computing 2019_, **Meng Ding et al.** 
* TR Decomposition [[Web]](https://qibinzhao.github.io/) [[Code]](https://qibinzhao.github.io/) [[PDF]](https://arxiv.org/pdf/1606.05535.pdf)
   * Tensor Ring Decomposition, _arXiv 2016_, **Qibin Zhao et al.**
* TRLRF [[Web]](https://qibinzhao.github.io/) [[Code]](https://github.com/yuanlonghao/TRLRF) [[PDF]](https://arxiv.org/abs/1809.02288)
   * Tensor Ring Decomposition with Rank Minimization on Latent Space: An Efficient Approach for Tensor Completion, _AAAI 2019_, **Longhao Yuan et al.**
* TN Rank [[Web]](https://arxiv.org/abs/1801.02662) [[Code]] [[PDF]](https://arxiv.org/pdf/1801.02662.pdf)
   * Tensor Network Ranks, _arXiv 2019_, **Ke Ye et al.**
* FCTN Decomposition [[Web]](https://yubangzheng.github.io/) [[Code]](https://yubangzheng.github.io/) [[PDF]](https://yubangzheng.github.io/papers/AAAI2021_FCTN_Decomposition_ybz.pdf)
   * Fully-Connected Tensor Network Decomposition and Its Application to Higher-Order Tensor Completion, _AAAI 2021_, **Yu-Bang Zheng et al.**
* NL-FCTN Decomposition [[Web]](https://zhaoxile.github.io/) [[Code]] [[PDF]](https://yubangzheng.github.io/papers/NL-FCTN-wjz.pdf)
   * Nonlocal Patch-Based Fully-Connected Tensor Network Decomposition for Remote Sensing Image Inpainting, _arXiv 2021_, **Wen-Jie Zheng et al.**
 



## Deep Learning
 * TT-LSTM [[Web]](https://www.dbs.ifi.lmu.de/~tresp/) [[Code]](https://github.com/Tuyki/TT_RNN) [[PDF]](http://proceedings.mlr.press/v70/yang17e/yang17e.pdf)
   * Tensor-Train Recurrent Neural Networks for Video Classification (ICML 2017), Yinchong Yang et al.
 * TT-Layer [[Web]](https://github.com/Bihaqo) [[Code]](https://github.com/Bihaqo/TensorNet) [[PDF]](https://papers.nips.cc/paper/5787-tensorizing-neural-networks.pdf)
   * Tensorizing Neural Networks (NIPS 2015), Alexander Novikov et al.



   
 ## Tensor Toolbox
 * Tensor Toolbox [[Web]](https://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html)
 * Tensorlab [[Web]](https://www.tensorlab.net/) 
 * TDALAB [[Web]](https://github.com/andrewssobral/TDALAB)  
 * TT-Toolbox  [[Web]](https://github.com/oseledets/TT-Toolbox) 
 

## Real-World Applications



## Commonly Used  Dataset
 * Videos [[Web]](http://trace.eas.asu.edu/yuv/) 
 * Multi-spectral Images [[Web]](http://www.cs.columbia.edu/CAVE/databases/multispectral/) 
 * Hyper-spectral Images [Web] 

## Commonly Used Image Quality Metrics
 * PSNR (Peak Signal-to-Noise Ratio) [[Wiki]](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) [[Matlab Code]](https://www.mathworks.com/help/images/ref/psnr.html) 
 * SSIM (Structural similarity) [[Wiki]](https://en.wikipedia.org/wiki/Structural_similarity) [[Matlab Code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) 




   
