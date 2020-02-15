# reproducible-tensor-completion-state-of-the-art
Collection of popular and reproducible tensor completion works.

Criteria: works must have codes available, and the reproducible results demonstrate state-of-the-art performances.


**This list is maintained by:**

**Xi-Le Zhao, Teng-Yu Ji, Tai-Xiang Jiang, Meng Ding, Yu-Bang Zheng** **[[Email]](https://zhaoxile.github.io/) (UESTC)**


## Excellent Review
#### Filter
 * NLM [[Web]](https://sites.google.com/site/shreyamsha/publications/image-denoising-based-on-nlfmt) [[Code]](https://www.mathworks.com/matlabcentral/fileexchange/44090-image-denoising-based-on-non-local-means-filter-and-its-method-noise-thresholding?focused=3806802&tab=function) [[PDF]](https://link.springer.com/article/10.1007/s11760-012-0389-y)
   * A non-local algorithm for image denoising (CVPR 05), Buades et al.
   * Image denoising based on non-local means filter and its method noise thresholding (SIVP2013), B. Kumar
 * BM3D [[Web]](http://www.cs.tut.fi/~foi/GCF-BM3D/) [[Code]](http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D.zip) [[PDF]](http://www.cs.tut.fi/~foi/GCF-BM3D/SPIE08_deblurring.pdf)
   * Image restoration by sparse 3D transform-domain collaborative filtering (SPIE Electronic Imaging 2008), Dabov et al.   
 * PID [[Web]](http://www.cgg.unibe.ch/publications/progressive-image-denoising) [[Code]](http://www.cgg.unibe.ch/publications/progressive-image-denoising/pid.zip) [[PDF]](http://www.cgg.unibe.ch/publications/2014/progressive-image-denoising/at_download/file)
   * Progressive Image Denoising (TIP 2014), C. Knaus et al.

## CANDECOMP/PARAFAC Decomposition
#### Filter
 * NLM [[Web]](https://sites.google.com/site/shreyamsha/publications/image-denoising-based-on-nlfmt) [[Code]](https://www.mathworks.com/matlabcentral/fileexchange/44090-image-denoising-based-on-non-local-means-filter-and-its-method-noise-thresholding?focused=3806802&tab=function) [[PDF]](https://link.springer.com/article/10.1007/s11760-012-0389-y)
   * A non-local algorithm for image denoising (CVPR 05), Buades et al.
   * Image denoising based on non-local means filter and its method noise thresholding (SIVP2013), B. Kumar
 * BM3D [[Web]](http://www.cs.tut.fi/~foi/GCF-BM3D/) [[Code]](http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D.zip) [[PDF]](http://www.cs.tut.fi/~foi/GCF-BM3D/SPIE08_deblurring.pdf)
   * Image restoration by sparse 3D transform-domain collaborative filtering (SPIE Electronic Imaging 2008), Dabov et al.   
 * PID [[Web]](http://www.cgg.unibe.ch/publications/progressive-image-denoising) [[Code]](http://www.cgg.unibe.ch/publications/progressive-image-denoising/pid.zip) [[PDF]](http://www.cgg.unibe.ch/publications/2014/progressive-image-denoising/at_download/file)
   * Progressive Image Denoising (TIP 2014), C. Knaus et al.

#### Tucker Decomposition
 * KSVD [[Web]](http://www.cs.technion.ac.il/~ronrubin/software.html) [[Code]](https://github.com/jbhuang0604/SelfSimSR/tree/master/Lib/KSVD) [[PDF]](http://www.egr.msu.edu/~aviyente/elad06.pdf)
   * Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries (TIP 2006), Elad et al.
 * LSSC [[Web]](https://lear.inrialpes.fr/people/mairal/) [[Code]](https://lear.inrialpes.fr/people/mairal/resources/denoise_ICCV09.tar.gz) [[PDF]](http://www.di.ens.fr/~fbach/iccv09_mairal.pdf)
   * Non-local Sparse Models for Image Restoration (ICCV 2009), Mairal et al.
 * NCSR [[Web]](http://www4.comp.polyu.edu.hk/~cslzhang/NCSR.htm) [[Code]](http://www4.comp.polyu.edu.hk/~cslzhang/code/NCSR.rar) [[PDF]](http://www4.comp.polyu.edu.hk/~cslzhang/paper/NCSR_TIP_final.pdf)
   * Nonlocally Centralized Sparse Representation for Image Restoration (TIP 2012), Dong et al.  
 * OCTOBOS [[Web]](http://transformlearning.csl.illinois.edu/projects/) [[Code]](https://github.com/wenbihan/octobos_IJCV2016) [[PDF]](http://transformlearning.csl.illinois.edu/assets/Sai/JournalPapers/SaiBihanIJCV2014OCTOBOS.pdf)
   * Structured Overcomplete Sparsifying Transform Learning with Convergence Guarantees and Applications (IJCV 2015), Wen et al. 
 * GSR [[Web]](https://jianzhang.tech/projects/GSR/) [[Code]](http://csjianzhang.github.io/codes/GSR_Code_Package_3.0.zip) [[PDF]](http://csjianzhang.github.io/papers/TIP2014_single.pdf)
   * Group-based Sparse Representation for Image Restoration (TIP 2014), Zhang et al.
 * TWSC [[Web]](https://github.com/csjunxu/TWSC-ECCV2018) [[Code]](https://github.com/csjunxu/TWSC-ECCV2018) [[PDF]](http://openaccess.thecvf.com/content_ECCV_2018/papers/XU_JUN_A_Trilateral_Weighted_ECCV_2018_paper.pdf)
   * A Trilateral Weighted Sparse Coding Scheme for Real-World Image Denoising (ECCV 2018), Xu et al.
  
#### Tensor Singular Value Decomposition
 * TRPCA [[Web]](https://canyilu.github.io/publications/) [[Code]](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) [[PDF]](https://arxiv.org/abs/1804.03728)
   * Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm (PAMI2018), Canyi Lu et al.
 * 3DTNN [[Web]](https://zhaoxile.github.io/) [[Code]](https://github.com/zhaoxile/Mixed-Noise-Removal-in-Hyperspectral-Image-via-Low-Fibered-Rank-Regularization) [[PDF]](https://ieeexplore.ieee.org/document/8854307)
* Mixed Noise Removal in Hyperspectral Image via Low-Fibered-Rank Regularization (IEEE TGRS2020), Yu-Bang Zheng et al.
* TNN [[Web]](http://www.ece.tufts.edu/~shuchin/) [[Code]]http://www.ece.tufts.edu/~shuchin/tensor_completion_and_rpca.zip) [[PDF]](https://www.zpascal.net/cvpr2014/Zhang_Novel_Methods_for_2014_CVPR_paper.pdf)
* Novel Methods for Multilinear Data Completion and De-noising Based on Tensor-SVD (CVPR2014), Zemin Zhang et al.
   
#### Tensor Train Decomposition
 *TMac_TT [[Web]](https://sites.google.com/site/jbengua/home) [[Code]](https://sites.google.com/site/jbengua/home/projects/efficient-tensor-completion-for-color-image-and-video-recovery-low-rank-tensor-train/TMacTT_Images.zip?attredirects=0&d=1) [[PDF]](https://www.researchgate.net/publication/303821165_Efficient_Tensor_Completion_for_Color_Image_and_Video_Recovery_Low-Rank_Tensor_Train)
   * Efficient Tensor Completion for Color Image and Video Recovery: Low-Rank Tensor Train (IEEE TIP2017), Johann A. Bengua et al.

   
#### Tensor Ring Decomposition
 * SF [[Web]](http://www.visinf.tu-darmstadt.de/vi_research/code/index.en.jsp#shrinkage_fields) [[Code]](https://github.com/uschmidt83/shrinkage-fields) [[PDF]](http://research.uweschmidt.org/pubs/cvpr14schmidt.pdf)
   * Shrinkage Fields for Effective Image Restoration (CVPR 2014), Schmidt et al.
 * TNRD [[Web]](http://www.icg.tugraz.at/Members/Chenyunjin/about-yunjin-chen) [[Code]](https://www.dropbox.com/s/8j6b880m6ddxtee/TNRD-Codes.zip?dl=0) [[PDF]](https://arxiv.org/pdf/1508.02848.pdf)
   * Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration (TPAMI 2016), Chen et al.
 * RED [[Web]](https://bitbucket.org/chhshen/image-denoising/) [[Code]](https://bitbucket.org/chhshen/image-denoising/) [[PDF]](https://arxiv.org/pdf/1603.09056.pdf)
   * Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections (NIPS2016), Mao et al.
 * DnCNN [[Web]](https://github.com/cszn/DnCNN) [[Code]](https://github.com/cszn/DnCNN) [[PDF]](https://arxiv.org/pdf/1608.03981v1.pdf)
   * Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (TIP2017), Zhang et al.
 * MemNet [[Web]](https://github.com/tyshiwo/MemNet) [[Code]](https://github.com/tyshiwo/MemNet) [[PDF]](http://cvlab.cse.msu.edu/pdfs/Image_Restoration%20using_Persistent_Memory_Network.pdf)
   * MemNet: A Persistent Memory Network for Image Restoration (ICCV2017), Tai et al.  
 * WIN [[Web]](https://github.com/cswin/WIN) [[Code]](https://github.com/cswin/WIN) [[PDF]](https://arxiv.org/pdf/1707.09135.pdf)
   * Learning Pixel-Distribution Prior with Wider Convolution for Image Denoising (Arxiv), Liu et al.    
 * F-W Net [[Web]](https://github.com/sunke123/FW-Net) [[Code]](https://github.com/sunke123/FW-Net) [[PDF]](https://arxiv.org/abs/1802.10252)
   * L_p-Norm Constrained Coding With Frank-Wolfe Network (Arxiv), Sun et al.
 * NLCNN [[Web]](https://cig.skoltech.ru/publications) [[Code]](https://github.com/cig-skoltech/NLNet) [[PDF]](http://www.skoltech.ru/app/data/uploads/sites/19/2017/06/1320.pdf)
   * Non-Local Color Image Denoising with Convolutional Neural Networks (CVPR 2017), Lefkimmiatis.
 * xUnit [[Web]](https://github.com/kligvasser/xUnit) [[Code]](https://github.com/kligvasser/xUnit) [[PDF]](https://arxiv.org/pdf/1711.06445.pdf)
   * xUnit: Learning a Spatial Activation Function for Efficient Image Restoration (Arxiv), Kligvasser et al.  
 * UDNet [[Web]](https://github.com/cig-skoltech/UDNet) [[Code]](https://github.com/cig-skoltech/UDNet) [[PDF]](https://arxiv.org/pdf/1711.07807.pdf)
   * Universal Denoising Networks : A Novel CNN Architecture for Image Denoising (CVPR 2018), Stamatios  Lefkimmiatis.   
 * Wavelet-CNN [[Web]](https://github.com/lpj0/MWCNN) [[Code]](https://github.com/lpj0/MWCNN) [[PDF]](https://arxiv.org/abs/1805.07071)
   * Multi-level Wavelet-CNN for Image Restoration (Arxiv), Liu et al.  
 * FFDNet [[Web]](https://github.com/cszn/FFDNet/) [[Code]](https://github.com/cszn/FFDNet/) [[PDF]](https://arxiv.org/abs/1710.04026)
   * FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising (TIP), Zhang et al.
 * FC-AIDE [[Web]](https://github.com/csm9493/FC-AIDE) [[Code]](https://github.com/csm9493/FC-AIDE) [[PDF]](https://arxiv.org/pdf/1807.07569.pdf)
   * Fully Convolutional Pixel Adaptive Image Denoiser (Arxiv), Cha et al.  
 * CBDNet [[Web]](https://github.com/GuoShi28/CBDNet) [[Code]](https://github.com/GuoShi28/CBDNet) [[PDF]](https://arxiv.org/pdf/1807.04686.pdf)
   * Toward Convolutional Blind Denoising of Real Photographs (Arxiv), Guo et al.  
 * UDN [[Web]](https://cig.skoltech.ru/publications) [[Code]](https://github.com/cig-skoltech/UDNet) [[PDF]](http://www.skoltech.ru/app/data/uploads/sites/19/2018/03/UDNet_CVPR2018.pdf)
   * Universal Denoising Networks- A Novel CNN Architecture for Image Denoising (CVPR 2018), Lefkimmiatis.     
 * N3 [[Web]](https://github.com/visinf/n3net) [[Code]](https://github.com/visinf/n3net) [[PDF]](https://arxiv.org/abs/1810.12575)
   * Neural Nearest Neighbors Networks (NIPS 2018), Plotz et al.  
 * NLRN [[Web]](https://github.com/Ding-Liu/NLRN) [[Code]](https://github.com/Ding-Liu/NLRN) [[PDF]](https://arxiv.org/pdf/1806.02919.pdf)
   * Non-Local Recurrent Network for Image Restoration (NIPS 2018), Liu et al.
 * RDN+ [[Web]](https://github.com/yulunzhang/RDN) [[Code]](https://github.com/yulunzhang/RDN) [[PDF]](https://arxiv.org/abs/1812.10477)
   * Residual Dense Network for Image Restoration (CVPR 2018), Zhang et al.
 * FOCNet [[Web]](https://github.com/hsijiaxidian/FOCNet) [[Code]](https://github.com/hsijiaxidian/FOCNet) [[PDF]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jia_FOCNet_A_Fractional_Optimal_Control_Network_for_Image_Denoising_CVPR_2019_paper.pdf)
   * FOCNet: A Fractional Optimal Control Network for Image Denoising (CVPR 2019), Jia et al.


## Deep Learning
#### Filter
 * NLM [[Web]](https://sites.google.com/site/shreyamsha/publications/image-denoising-based-on-nlfmt) [[Code]](https://www.mathworks.com/matlabcentral/fileexchange/44090-image-denoising-based-on-non-local-means-filter-and-its-method-noise-thresholding?focused=3806802&tab=function) [[PDF]](https://link.springer.com/article/10.1007/s11760-012-0389-y)
   * A non-local algorithm for image denoising (CVPR 05), Buades et al.
   * Image denoising based on non-local means filter and its method noise thresholding (SIVP2013), B. Kumar
 * BM3D [[Web]](http://www.cs.tut.fi/~foi/GCF-BM3D/) [[Code]](http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D.zip) [[PDF]](http://www.cs.tut.fi/~foi/GCF-BM3D/SPIE08_deblurring.pdf)
   * Image restoration by sparse 3D transform-domain collaborative filtering (SPIE Electronic Imaging 2008), Dabov et al.   
 * PID [[Web]](http://www.cgg.unibe.ch/publications/progressive-image-denoising) [[Code]](http://www.cgg.unibe.ch/publications/progressive-image-denoising/pid.zip) [[PDF]](http://www.cgg.unibe.ch/publications/2014/progressive-image-denoising/at_download/file)
   * Progressive Image Denoising (TIP 2014), C. Knaus et al.

#### Real-World Applications
 * SINLE [[PDF]](http://www.ok.sc.e.titech.ac.jp/res/NLE/TIP2013-noise-level-estimation06607209.pdf) [[Code]](https://www.mathworks.com/matlabcentral/fileexchange/36921-noise-level-estimation-from-a-single-image) [[Slides]](https://wwwpub.zih.tu-dresden.de/~hh3/Hauptsem/SS16/noise.pdf)
   * Single-image Noise Level Estimation for Blind Denoising (TIP 2014), Liu et al.



   
#### Commonly Used  Dataset
 * Kodak [[Web]](http://r0k.us/graphics/kodak/)
 * USC SIPI-Misc [[Web]](http://sipi.usc.edu/database/database.php?volume=misc) 
 * BSD [[Web]](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)  

#### Commonly Used Image Quality Metrics
 * PSNR (Peak Signal-to-Noise Ratio) [[Wiki]](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) [[Matlab Code]](https://www.mathworks.com/help/images/ref/psnr.html) [[Python Code]](https://github.com/aizvorski/video-quality)
 * SSIM (Structural similarity) [[Wiki]](https://en.wikipedia.org/wiki/Structural_similarity) [[Matlab Code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[Python Code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
 * NIQE (Naturalness Image Quality Evaluator) [[Web]](http://live.ece.utexas.edu/research/Quality/nrqa.htm) [[Matlab Code]](http://live.ece.utexas.edu/research/Quality/nrqa.htm) [[Python Code]](https://github.com/aizvorski/video-quality/blob/master/niqe.py)



   
