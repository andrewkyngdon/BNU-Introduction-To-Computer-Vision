# BNU Introduction To Deep Learning Workshop - Computer Vision

The MNIST and CIFAR-10 datasets are the "Hello World" of Deep Learning models for Computer Vision. This is a repository of models trained as part of an introductory workshop in Deep Learning for social science students at Beijing Normal University, Beijing, China.

All models were trained with [Caffe for Windows](https://github.com/BVLC/caffe/tree/windows) using an NVIDIA GTX 1060 GPU, CUDA 8.0 and cuDNN 6.0. The official Caffe repo for any supported OS can be used to test these models without modification.

### Usage

Clone this repo to the desired location. Then download each of these test datasets into the relevant directory:

Model|Download
:---:|:---:
MNIST| [tst_mnist.hdf5](https://drive.google.com/open?id=1TuPZ2SMl0CHsf24I2KDk5D3VELzyhHKJ)
CIFAR-10| [test_lmdb_cifar10](https://drive.google.com/open?id=1fm5klPCpjdGBBRZVvdQE_qW4u_3VzKfm)
LSI-FIR| [test_lmdb_lsi](https://drive.google.com/open?id=1g8Zcw_BbwKetZ22f-V6IjZ19MwhmU1F4)

At the command prompt, navigate to the relevant directory (e.g., CIFAR_10) and type:

```
caffe test -model <the_model_to_test>.prototxt -weights <the_model_to_test>.caffemodel -iterations 100 -gpu 0
```
Test with your CPU by omitting the GPU flag. For the LSI-FIR model, set the number of iterations to 140.

### MNIST

The trained MNIST models consist of logistic regression, shallow feed-forward nets of one and two hidden layers; and the convolutional Caffe LeNet architecture that ships with Caffe.

Test set Top-1 accuracy rates and cross-entropy losses were as follows:

Model|Top-1|Loss
:---:|:---:|:---:
Logistic Regression|90.59|0.328
One Hidden Layer|97.47|0.087
Two Hidden Layers|97.80|0.069
Caffe LeNet|99.13|0.029

**MNIST Data Pre-processing**
- Standard greyscale image rescaling **scale: 0.00390625**
- Random shifts between 3 and 5 pixels inclusive
- Random clockwise and counter-clockwise rotations

### CIFAR-10

Models trained were [SqueezeNet Residual](https://github.com/songhan/SqueezeNet-Residual) with [swish neurons](https://arxiv.org/abs/1710.05941) and [DenseNet L=40 k=12](https://github.com/liuzhuang13/DenseNetCaffe).

Model|Top-1|Loss
:---:|:---:|:---:
SqueezeNetRes|81.34|0.565
DenseNet-40-12|90.05|0.317

**CIFAR-10 Data Pre-preprocessing**
- BGR mean values **[111.801,120.039,122.46]** were subtracted
- 30 pixel crops **crop_size: 30**
- Random flips **mirror: true**
- Random clockwise and counter-clockwise rotations

**Notes**
- CIFAR-10 requires heavy data augmentation to achieve state-of-the-art. The preprocessing performed here was light and done for pedagogical purposes.
- SqueezeNet Residual was initialised using these [weights trained on ImageNet](https://github.com/songhan/SqueezeNet-Residual).
- DenseNet-40-12 can achieve 92.91% Top-1 accuracy out of the box on CIFAR-10 without data augmentation, [according to its authors](https://github.com/liuzhuang13/DenseNetCaffe). However, considerably longer training appears to be required than what was done here.
- A Jupyter Notebook `Visualise CIFAR10.ipynb` can be used to visualise test image data, convolutional filters and neurons. Global pooling layer activations can be visualised using the dimensionality reduction technique of [t-SNE](https://lvdmaaten.github.io/tsne/). Simply type `jupyter notebook` at the command prompt and open the file in your browser. Edit as necessary.

### LSI Far Infrared Pedestrian Database - Classification

This [dataset](https://portal.uc3m.es/portal/page/portal/dpto_ing_sistemas_automatica/investigacion/IntelligentSystemsLab/research/InfraredDataset) consists of far infrared images of pedestrians taken by a vehicle mounted camera.

A modified variant of the Caffe LeNet architecture was trained. Because of class imbalance, precision, recall and F1 statistics are provided. These were obtained by modifying [this Python Caffe layer](https://github.com/gcucurull/caffe-conf-matrix) to include relevant functions from the `sklearn.metrics` Python library.

Model|Top-1|Loss|F1|Precision|Recall|Architecture
:---:|:---:|:---:|:---:|:---:|:---:|:---:
LSI-FIR|98.88|0.0362|0.9737|0.9715|0.9759| [netscope](http://ethereon.github.io/netscope/#/gist/214b2730a2af70a81aedcc4be5fd18fc)

**LSI Data Pre-preprocessing**
- Standard greyscale image rescaling **scale: 0.00390625**
- Training set class imbalance was reduced through random undersampling of the "pedestrian absent" image data for an approximate 34/66 split.
