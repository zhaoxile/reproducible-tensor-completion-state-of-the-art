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
 * KBR [[Web]](http://gr.xjtu.edu.cn/web/dymeng/1;jsessionid=F03A6AE30867A1EE7DE9D577DD4E253D) [[Code]](https://github.com/XieQi2015/KBR-TC-and-RPCA) [[PDF]](https://ieeexplore.ieee.org/iel7/34/4359286/08000407.pdf)
   * Kronecker-Basis-Representation Based Tensor Sparsity and Its Applications to Tensor Recovery (PAMI 2017), Qi Xie et al.

  
#### Tensor Singular Value Decomposition
 * TRPCA [[Web]](https://canyilu.github.io/publications/) [[Code]](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) [[PDF]](https://arxiv.org/abs/1804.03728)
   * Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm (PAMI2018), Canyi Lu et al.
 * 3DTNN [[Web]](https://zhaoxile.github.io/) [[Code]](https://github.com/zhaoxile/Mixed-Noise-Removal-in-Hyperspectral-Image-via-Low-Fibered-Rank-Regularization) [[PDF]](https://zhaoxile.github.io/paper/2020/Mixed%20Noise%20Removal%20in%20Hyperspectral%20Image%20via%20Low-Fibered-Rank%20Regularization.pdf)
   * Mixed Noise Removal in Hyperspectral Image via Low-Fibered-Rank Regularization (IEEE TGRS2020), Yu-Bang Zheng et al.
* PSTNN [[Web]](https://sites.google.com/view/taixiangjiang/) [[Code]](https://github.com/zhaoxile/Multi-dimensional-imaging-data-recovery-via-minimizing-the-partial-sum-of-tubal-nuclear-norm) [[PDF]](https://zhaoxile.github.io/paper/2019/Multi-dimensional%20imaging%20data%20recovery%20via%20minimizing%20the%20partial%20sum%20of%20tubal%20nuclear%20norm.pdf)
   * Multi-dimensional imaging data recovery via minimizing the partial sum of tubal nuclear norm
 (JCAM2020), Tai-Xiang Jiang et al. 
* TNN [[Web]](http://www.ece.tufts.edu/~shuchin/) [[Code]](http://www.ece.tufts.edu/~shuchin/tensor_completion_and_rpca.zip) [[PDF]](https://www.zpascal.net/cvpr2014/Zhang_Novel_Methods_for_2014_CVPR_paper.pdf)
   * Novel Methods for Multilinear Data Completion and De-noising Based on Tensor-SVD (CVPR2014), Zemin Zhang et al.
   
#### Tensor Train Decomposition
 *TMac_TT [[Web]](https://sites.google.com/site/jbengua/home) [[Code]](https://sites.google.com/site/jbengua/home/projects/efficient-tensor-completion-for-color-image-and-video-recovery-low-rank-tensor-train/TMacTT_Images.zip?attredirects=0&d=1) [[PDF]](https://www.researchgate.net/publication/303821165_Efficient_Tensor_Completion_for_Color_Image_and_Video_Recovery_Low-Rank_Tensor_Train)
   * Efficient Tensor Completion for Color Image and Video Recovery: Low-Rank Tensor Train (IEEE TIP2017), Johann A. Bengua et al.
   
*MF-TTTV [[Web]](https://zhaoxile.github.io/) [[Code]](https://github.com/zhaoxile) [[PDF]](https://zhaoxile.github.io/paper/2019/Low-Rank%20Tensor%20Completion%20Using%20Matrix%20Factorization%20Based%20on%20Tensor%20Train%20Rank%20and%20Total%20Variation.pdf)
   * Low-Rank Tensor Completion Using Matrix Factorization Based on Tensor Train Rank and Total Variation (JSC2019), Meng Ding et al.
   
#### Tensor Ring Decomposition
 * SF [[Web]](http://www.visinf.tu-darmstadt.de/vi_research/code/index.en.jsp#shrinkage_fields) [[Code]](https://github.com/uschmidt83/shrinkage-fields) [[PDF]](http://research.uweschmidt.org/pubs/cvpr14schmidt.pdf)
   * Shrinkage Fields for Effective Image Restoration (CVPR 2014), Schmidt et al.



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



   
