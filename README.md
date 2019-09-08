Usage: `python grad-cam.py --image-path <path_to_image>`

To use with CUDA:
`python grad-cam.py --image-path <path_to_image> --use-cuda`

This above understands English should be able to understand how to use, I just changed the original vgg19 network into imagenet pre-trained resnet50, in fact, for any processing of pictures can still be used, but we are doing The video is very troublesome, because the network has more one-dimensional time dimension, which makes me a headache. Therefore, although I have changed this thing, I don’t have any sense of accomplishment. Let me use it for everyone who wants to use the resnet50 network to test the cam map.

##note

The default IMAGE_PATH above is already `./examples`


[Original link] (https://github.com/jacobgil/pytorch-grad-cam)


## Follow-up instructions
After two days of research, I found out that this cam is a simple feature that combines features into our original image. In fact, if the research is not very detailed, you don't need to understand the principle. Because the middle of a series of mathematical processes is actually to mention the features, we will combine this feature with the original image. Say a thousand 10,000. In fact, as long as we can extract features in the network, and then save the features, then we can use Opencv to combine the feature map and the original image can also be done. (Currently this path should be very traditional, stupid, but very simple to implement.)
If there is time to study again, it may be updated like this, but I feel that this work will be done here. After all, the visualization of video frames can be output directly at any position in the big project. Made it so troublesome.
At present, it can be confirmed that the first paragraph of the practice is theoretically no problem. The local show before the show_cam_on_image using the project and the two images with the opencv stitching, the two pictures print effect is not bad. (The opencv parameter may need to be fine-tuned. This varies from person to person depending on the individual situation.) If you really want to follow the 2d idea, it is possible that the mapping effect will be poor due to the deviation of the time dimension when setting up the network. (Because the video frames required by our data preprocessing are generally representative, so the frame and the frame must be guaranteed to be visible to the naked eye.)
## Thought Process
1. First solve the problem of extracting features. This is actually based on the built-in functions of the existing framework, and the features extracted are mainly filtered from the back of the conv layer. For the 3d network, it really exists because of the time dimension. At first, there is confusion about extracting features here. But simply think about it, as long as you take out each of the saved pictures in the time dimension and then print out the features corresponding to each picture and conv.
2. Extracting good features There are many ways to do this kind of flattening. I have mentioned above. If you continue to apply this project, it may happen that the corresponding features cannot be mapped to the original image. It really needs us to see the last distribution of the feature distribution mapped to the keyframe. (That is to say, we can save the feature first, then map the obtained features one by one to each keyframe, and select the best one with the naked eye, which is not a practice.)
## to sum up
It is true that for video frame processing, it is necessary to consider the final problem with different dimensional gradients. However, the cam can still make the features according to the code segment corresponding to the project, and then combine the corresponding cam map and the original clips according to the stitching method we have previously mentioned.
