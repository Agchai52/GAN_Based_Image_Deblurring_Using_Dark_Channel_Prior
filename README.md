# GAN Based Image Deblurring Using Dark Channel Prior

A conditional general adversarial network (GAN) is proposed for image deblurring problem. We concentrate on tailoring the network for image deblurring instead of just applying GAN on deblurring problem. Motivated by that, dark channel prior is carefully picked to be incorporated into loss function for network training. To make it more compatible to neuron networks, we discard its original indifferentiable form and adopt L2 norm instead. On both synthetic datasets and noisy natural images, the proposed network shows improved deblurring performance and robustness to image noise qualitatively and quantitatively. Additionally, compared to the existing end-to-end deblurring networks, our network structure is light-weight, which ensures less training and testing time. 


## Prepare train and test datasets

We use the datasets from DeepDeblur and DeblurGAN

Use image_scale.m to generate folders blur_all_train, sharp_all_train, blur_all_test, sharp_all_test

python combine_A_B.py --fold_A ./GOPRO_Large/blur_all_train --fold_B ./GOPRO_Large/sharp_all_train --fold_AB ./datasets/train

python combine_A_B.py --fold_A ./GOPRO_Large/blur_all_test --fold_B ./GOPRO_Large/sharp_all_test --fold_AB ./datasets/test


## Train 

python main.py --phase train --gpu 0 --add_noise False

python plot_loss.py 


## Test and save test results

python main.py --phase test --fine_size 360


## Test and only compute psnr of test results

python main.py --phase test --fine_size 360 --test_psnr True

## Citation
@article {DarkChannelGAN:2019,
title = "GAN based image deblurring using dark channel prior",
journal = "Fast track article for IS\&T International Symposium on Electronic Imaging 2019: Computational Imaging XVII proceedings.",
year = "2019",
itemtype = "ARTICLE",
issn = "2470-1173",
pages = {136-1--136-5},
eissn = "2470-1173",
doi = "doi:10.2352/ISSN.2470-1173.2019.13.COIMG-136",
author = "Zhang, Shuang and Zhen, Ada and Stevenson, Robert L.",
}
