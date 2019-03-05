## Grad-CAM implementation in Pytorch ##

### What makes the network think the image label is 'pug, pug-dog' and 'tabby, tabby cat':
![Dog](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors torch implementation: https://github.com/ramprs/grad-cam

My Keras implementation: https://github.com/jacobgil/keras-grad-cam


----------


This uses VGG19 from torchvision. It will be downloaded when used for the first time.

The code can be modified to work with any model.
However the VGG models in torchvision have features/classifier methods for the convolutional part of the network, and the fully connected part.
This code assumes that the model passed supports these two methods.


----------


Usage: `python grad-cam.py --image-path <path_to_image>`

To use with CUDA:
`python grad-cam.py --image-path <path_to_image> --use-cuda`


这个上面的懂点英文都应该看得懂怎么用，我这个只不过把原始的vgg19网络改成了imagenet预训练的resnet50而已，实际上对于任意的处理图片的还是可以用的，但是我们是做视频的，就搞得很麻烦，因为网络多一维时间维，搞得我很头痛，虽然改出来了这个东西但是并没有什么成就感。