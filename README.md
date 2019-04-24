Usage: `python grad-cam.py --image-path <path_to_image>`

To use with CUDA:
`python grad-cam.py --image-path <path_to_image> --use-cuda`


这个上面的懂点英文都应该看得懂怎么用，我这个只不过把原始的vgg19网络改成了imagenet预训练的resnet50而已，实际上对于任意的处理图片的还是可以用的，但是我们是做视频的，就搞得很麻烦，因为网络多一维时间维，搞得我很头痛。因此虽然改出来了这个东西但是并没有什么成就感，放出来给各位想用resnet50网络来测试cam图的各位用吧。


## 注意

上面默认的image_path已经是./examples了


[原始链接](https://github.com/jacobgil/pytorch-grad-cam)


虽然看的还是英文，但是什么时候中文才能普及全球呢.....
