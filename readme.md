## ICCV 2019 Workshop & Challenge on Computer Vision for Wildlife Conservation (CVWC)


### Recent Updates
[2019.8.3]


### Dependencies
- python == 3.6
- torch == 0.4.1
- torchvision == 0.2.1


### Solution  
https://github.com/liuning-scu-cn/AmurTigerReID


### Prepare Data  
- Train data:

Please download from https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_reid_train.tar.gz

- Train Annotations:

Please download from https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_anno_reid_train.tar.gz

- Test data:

Please download from https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_reid_test.tar.gz

- Val data:

Please download from ...


### Competition Dataset  
train_set &ensp;&ensp;=> &ensp;&ensp;./dataload/dataloader.py  
test_set &ensp;&ensp;&ensp;=> &ensp;&ensp;./dataload/dataloader.py


### Train  
in train.py, finetune_tiger_cnn5.py, and finetune_tiger_cnn8.py. 


### Test  
in test.py  


### Thanks
- [CVWC 2019 Organizing Team](https://cvwc2019.github.io/)
- [EvalAI Team](https://evalai.cloudcv.org)
- [SiChuan University BRL](http://www.scubrl.org/index)

###### If you have any problems, please contact [BRL](http://www.scubrl.org/index) or email to 2742229056@qq.com.
