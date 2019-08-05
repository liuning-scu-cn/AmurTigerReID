## ICCV 2019 Workshop & Challenge on Computer Vision for Wildlife Conservation (CVWC)


### Recent Updates
[2019.8.4]

### Note
We participate in the Plain Re-ID Track. Our solution uses SE-ResNet50 model as backbone which was pre-trained by ILSVRC. In addition, we design two complementary network branches to learn multiple discriminative features. We use multi-task learning strategy to supervise the model training. Finally, we fine-tune the model with triplet loss. The Re-ID results are obtained based on the fusion of the learned multiple features.


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

If you want to run finetune_tiger_cnn5.py, you firstly need to train tiger_cnn1 model.

If you want to run finetune_tiger_cnn8.py, you firstly need to train tiger_cnn3 model.

### Test  
in test.py  

### Model
If you want to get our all models and logs, please download from [https://pan.baidu.com/s/11RslAFW9g7kS8I4IVZC_Iw](https://pan.baidu.com/s/11RslAFW9g7kS8I4IVZC_Iw), and passward code is ```7iic```.
Then you can save it to ./model.

### Thanks
- [CVWC 2019 Organizing Team](https://cvwc2019.github.io/)
- [EvalAI Team](https://evalai.cloudcv.org)
- [SiChuan University BRL](http://www.scubrl.org/index)

###### If you have any problems, please contact [BRL](http://www.scubrl.org/index) or email to 2742229056@qq.com.
