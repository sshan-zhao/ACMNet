# ACMNet
This the Pytorch implementation of our work on depth completion.

**S. Zhao, G. Gong, H. Fu and D. Tao. Adaptive Context-Aware Multi-Modal Network for Depth Completion. [PAPER]()**


## Environment
1. Python 3.6
2. PyTorch 1.2.0
3. CUDA 10.0
4. Ubuntu 16.04
5. Opencv-python
6. pip install pointlib/.

## Datasets
[KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)

Prepare the dataset according to the datalists (*.txt in [datasets](./datasets))
```
datasets
|----kitti 
    |----depth_selection 
        |----val_selection_cropped
            |----...
        |----test_depth_completion_anonymous   
            |----...     
    |----rgb     
        |----2011_09_26
        |----...  
    |----train  
        |----2011_09_26_drive_0001_sync
        |----...      
```

## Training 
We will release the training code after the peer-review process.

## Test
run
```
bash run_eval.sh
```
Note that, currently we only release the pretrained model with 32 channels.

## Contact
Shanshan Zhao: szha4333@uni.sydney.edu.au or sshan.zhao00@gmail.com
