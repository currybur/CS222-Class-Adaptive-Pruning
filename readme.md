# Class Adaptive Pruning of CNN  
**A shallow insight into convolution neural network**  
***
算法设计与分析课的大作业。这课我学得不好😪，英文授课以及我总爱走神，课后花时间也不够。大作业和算法没什么联系，如标题所言，主要是理解卷积神经网络结构和粗浅了解一些神经网络可视化。

更新：组员毕设中重新研究了这个工作，发现CAPTOR论文中求激活梯度似乎对象搞错了，应当是对卷积核权重求导而非特征图求导。[他的仓库](https://github.com/david990917/Model-Pruning-in-Deep-Learning)。
***
## 剪枝与类适应剪枝
对神经网络剪枝使得其轻量化，便于部署到网络边缘设备上，这在今天移动互联网蓬勃发展的背景下尤其重要。实验表明当今流行的神经网络模型中许多结构是冗余的，去掉一些对准确率没什么影响，发展出了许多剪枝方法。大作业主要是复现***CAPTOR: A Class Adaptive Filter Pruning Framework for Convolutional Neural Networks in Mobile Applications***[<sup>1</sup>](#refer-anchor-1)，通过  

1. Activation Maximization[<sup>2</sup>](#refer-anchor-2)可视化卷积核的输出并且聚类
2. 类别标签对卷积核的激活值的梯度FIP

这两个标准来衡量卷积核对各类别标签的贡献度，从而将一个有许多目标类别的大分类网络简化成有较少目标类别的小网络，而保持对于这些目标类别的准确性。  
这种做法是有道理的，因为实际应用场景特别是移动终端下，标准分类网络中大部分良好训练的类别标签并没有用上，或者是某种单一用途的终端不需要大量的目标类别。当然针对需要的类别重新训练小模型，但通过剪枝可以省去重新训练的时间，直接对训练好的模型动刀，精简掉大量的结构可以提升大量的计算速度，而精度与大模型接近甚至更高（可能避免了一些潜在的干扰选项）。

## 实验和结果
用了cifar 10 数据集[<sup>3</sup>](#refer-anchor-3)训练VGG-19，看看AM效果（不愧被称为DeepDream，你认得出来图里都是什么品种吗👽）。  

<img src="https://github.com/currybur/CS222-Class-Adaptive-Pruning/raw/master/am_img/1.png" width="15%"> <img src="https://github.com/currybur/CS222-Class-Adaptive-Pruning/raw/master/am_img/2.png" width="15%"> <img src="https://github.com/currybur/CS222-Class-Adaptive-Pruning/raw/master/am_img/3.png" width="15%"> <img src="https://github.com/currybur/CS222-Class-Adaptive-Pruning/raw/master/am_img/4.png" width="15%">

原论文主要是AM聚类+梯度贴标签，然后删掉相应类别的cluster里的卷积核，就可以快速剪枝了，称为cluster level pruning。我们考虑了一个卷积核可能对多个类别有较大贡献的情况，提出filter level pruning，相当于更细粒度了。  
然后基于用cifar-10训练的VGG-19简单测试了一下两个方法。然而很可惜，我们的剪枝效果其实不太好，主要在于原文提出的用梯度来贴标签我们做出来标签似乎并不准确，比如一个卷积核的AM看起来是鸟，而梯度贡献最大的却是车，可能实现方法有问题，不过作业交完就懒得再研究了🙃。

## 总结
虽然吐槽大作业炼丹，不过搞清楚了pytorch和CNN的基本知识对后续更多的炼丹大作业还是有帮助的🤣。玩炼丹其实是有趣的，可是专业课上还是希望学到更多传统CS的基本功。  


## 参考
<div id="refer-anchor-1"></div>

- [1] [Qin Z, Yu F, Liu C, et al. CAPTOR: a class adaptive filter pruning framework for convolutional neural networks in mobile applications[C]//Proceedings of the 24th Asia and South Pacific Design Automation Conference. 2019: 444-449.](https://dl.acm.org/doi/abs/10.1145/3287624.3287643)
- [2] [Yosinski J, Clune J, Nguyen A, et al. Understanding neural networks through deep visualization[J]. arXiv preprint arXiv:1506.06579, 2015.](https://arxiv.org/abs/1506.06579)
- [3] [Krizhevsky A, Hinton G. Learning multiple layers of features from tiny images[J]. 2009.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf)
