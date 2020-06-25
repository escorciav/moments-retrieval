# Ideas for CVPR21

## Areas:
1. Self supervised pretraining 
2. Loss improvement 
3. Model novelty
4. Data augmentation

## High priority
- Change implementation of negative utilization in the batch to speed up the training. Each sample is a negative for anoter one.
- I3D features. [Repo](https://github.com/piergiaj/pytorch-i3d)
- HowTo1M features for visual branch. I need to get the network weights.

## 1. Self supervised pretraining 
- Similar to several works but inspired by HowTo100M [link](https://arxiv.org/pdf/1906.03327.pdf).
- **TODO:** List more works from CVPR20.

## 2. Loss improvement
- Test MIL-NCE [link](https://arxiv.org/pdf/1912.06430.pdf) without chamfer distance. Same formulation as the paper.
- Extend MIL-NCE loss to use additional term coming from Angular Loss [link](http://research.baidu.com/Public/uploads/5acc20706a719.pdf). Requires to solve previous step or adapt angular loss to use chamfer distance. 
- Check more recent loss [link1](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)/[link2]().

------------------

- **Differently**: Check if we can learn how to rank with the paper [link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Rolinek_Optimizing_Rank-Based_Metrics_With_Blackbox_Differentiation_CVPR_2020_paper.pdf)

## 3. Model novelty
- **IDEA:** Since in the second stage retrieval we are already breaking the separability principle. Why don't we derive a more complex model that jointly embed the video and language to improve performance on the retrieval of the second stage?
- Better features? Pixel bert? X3D
- Can we learn a 3d network end to end?
- Transformers for language
- GCN for video.



## 4. Data augmentation
- Use this repo to create captions [Repo](https://github.com/v-iashin/BMT)/[Paper](https://arxiv.org/pdf/2005.08271.pdf)/[Nice Website](https://v-iashin.github.io/bmt.html)
- Use the extracted audio features, for previous point, as additional input to my network.


## Low priority/simple test:
- What happens if we increase the embedding size?
- Check if regularizers on top of embedding norm let us train with language augmented data.
- Test if replacing ReLU with sine function changes anything [link](https://arxiv.org/pdf/2006.09661.pdf) or Hermite Polynomial [link](https://arxiv.org/abs/1909.05479)
-Better investigate/keep track  of mining strategy. At some point it might start to work. 
- Motion features (use an out of the shelf network for motion estimation. It's an additional modalit as input).
- SPS new optimizer [Paper](https://arxiv.org/abs/2002.10542)/[Repo](https://github.com/IssamLaradji/sps).

