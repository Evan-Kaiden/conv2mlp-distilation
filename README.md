Exploring Whether a Non-Convolutional Model Can Learn Spatial Features via Distillation

# Results:
After training a small CNN on CIFAR-100 that achieved a test accuracy of 44%, I applied distillation techniques to train a Multi-Layer Perceptron (MLP) with no convolutional layers but an equivalent input-output structure. This distilled MLP achieved a test accuracy of 30%. In contrast, training the same MLP architecture from scratch without distillation resulted in only 20% test accuracy. These results suggest that the distillation method is effective and that the MLP is, to some extent, able to learn spatial features through knowledge transfer from the CNN
