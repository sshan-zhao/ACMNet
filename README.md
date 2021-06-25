# ACMNet
This is the Pytorch implementation of our work on depth completion.

**S. Zhao, M. Gong, H. Fu and D. Tao. Adaptive Context-Aware Multi-Modal Network for Depth Completion. (IEEE Trans. Image Process.) [Arxiv](https://arxiv.org/pdf/2008.10833.pdf)(Early Version) [IEEE](https://ieeexplore.ieee.org/abstract/document/9440471/)(Final Version)**


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
    |----val   
        |----2011_09_26_drive_0002_sync
        |----...
```

## Training 
run
```
bash run_train.sh
```

## Test
run
```
bash run_eval.sh (sval.txt for selected_validation, val for validation) or bash run_test.sh (for submission)
```

## Citation
```
@article{zhao2021adaptive,
  title={Adaptive context-aware multi-modal network for depth completion},
  author={Zhao, Shanshan and Gong, Mingming and Fu, Huan and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```
## Contact
Shanshan Zhao: szha4333@uni.sydney.edu.au or sshan.zhao00@gmail.com
