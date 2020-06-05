
AM implementation refer to: https://github.com/Sujit27/ActivationMaximization. 

Download pretrained model: https://jbox.sjtu.edu.cn/l/cuB1R0
## File list:
> * deepDream.py: implement AM and gradient computation.
> * main.py: run AM to get feature interest patterns.  
> * label_filters.py: sample images and calculate the class activation of filters and clusters.
> * test.py: prune filters on cluster-level as well as filter-level.
> * utils/k-means.py: cluster feature interest patterns.
> * utils/index.py: formulate the cluster results.
> * utils/extract_img.py: formulate and save images from torchvision datasets of cifar-10.
> * index/yils.json: the cluster-label table, used to do cluster-level pruning.  
> * index/filter_priority.json: the filter-label priority table, used to do filter-level prunig.  

To get feature interest patterns of different layers, change the network structure in deepDream.py, and specify the wanted filter numbers in main.py.
* Run utils/k-means.py to cluster these feature interest patterns.
* Run utils/index.py with specified path to get a simplified cluster index.
* Run label_filters.py to get cluster-label table and filter-label table.
* Run test.py with specified path and label to test accuracy and number of filters.
