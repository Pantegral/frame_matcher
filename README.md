# frame_matcher
Very simple implementation of potential video editing tool.

Idea is to find mask or outline of primary subject in a given frame, and then find the frames within a target video wose primary subject masks  most closely match that original mask.

Could be used to help with match cuts when video editing.

Uses pretrained yolonetv3 (Original paper: https://arxiv.org/abs/1804.02767) available from gluoncv's model zoo.

Currently would only advise very short video clips, but with a couple of changes would be much faster.
    1: Obviously changing context to gpu would be much faster, easily reaches 20-30fps. Just wrap all forward passes in with mx.Context(mx.gpu(0)):
    2: Could reduce sample rate by checking every x number of frames.
    
Currently would caution that the frames in the returned list are in descending order of the intersection over union of the highest class confidence score object in  the frame with the target. This will frequently not be the primary subject of the frame if there are multiple objects, so careful with busy frames. The network used is also trained on a limited set of class labels, and objects not within the labels defined in the Pascal VOC dataset will not work well.
    

To use just supply video path and frame path with your desired file_paths. Written in Jupyter Notebook, if desired as a .py you'll just need to add save functions for the output frames, or just print the video time at which the frame is found. The frame_match function of the frame_matcher is the only function required, so just pass it your frame_path and video_path, and it will return a tuple of (intersection over union with target, the image array, the list of all bounding boxes of all objects within the frame, the confidence scores of those bounding boxes, the class of those bounding boxes, and the time_stamp of the frame). 

Hopefully someone one day finds this maybe mildly useful.

