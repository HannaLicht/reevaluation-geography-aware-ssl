### Abstract:
Foundation models, trained on large datasets in a self-supervised manner, have become useful tools in artificial intelligence, 
especially in domains with abundant unlabeled data but scarce labeled data, such as remote sensing. This study re-evaluates the 
MoCo-v2 framework and compares it with a geography-aware contrastive learning approach introduced by Ayush et al. in the paper 
”Geography-Aware Self-Supervised Learning” [1]. This approach leverages spatio-temporal structures in remote-sensing data and 
combines it with a geo-location classification pre-text task. I use the BigEarthNet dataset to evaulate the expressivness of 
the models’ feature representations in several experiments. The experiments demonstrate that integrating metadata, particularly 
geo-location information, into foundation models is likely to improve performance in multi-label classification tasks.

[1] Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, and Stefano Ermon. Geography-aware 
    self-supervised learning. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10161–10170, 2021.
