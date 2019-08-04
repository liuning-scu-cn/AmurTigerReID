## Training Model

### Recent Updates
[2019.8.1]

If you want to get our all models and logs, please download from [https://pan.baidu.com/s/11RslAFW9g7kS8I4IVZC_Iw](https://pan.baidu.com/s/11RslAFW9g7kS8I4IVZC_Iw), and passward code is ```7iic```.
Then you can save it to ./model.

### Model Test

| Model-Name| mAP(single) | top@1(single) | top@5(single) | mAP(cross) | top@1(cross) | top@5(cross) |
| :-------: | :---------: | :-----------: | :-----------: | :--------: | :----------: | :----------: |
| official-baseline  | 71.4        | 86.6%         | 95.4%         | 48.1       | 79.4%        | 93.7%        |
| tiger_cnn1         | 83.9        | 92.0%         | 94.5%         | 60.9       | **94.8%**    |   96.5%      |
| tiger_cnn2         | 89.0        | 98.2%         | 98.5%         | 59.0       | 86.2%        | 95.4%        |
| tiger_cnn3         | 88.7        | 97.4%         | **98.8%**     | 57.4       | 86.8%        | 92.0%        |
| tiger_cnn4         |  -          |  -            |  -            |  -         |  -           |  -           |
| tiger_cnn5         | 90.2        | 97.1%         | 98.2%         | 60.7       | 89.7%        | 96.5%        |
| tiger_cnn6         | 87.8        | 95.4%         | 98.2%         | 58.4       | 89.7%        | 94.8%        |
| tiger_cnn7         | 87.7        | 95.7%         | 98.0%         | 58.0       | 88.5%        | 92.0%        |
| tiger_cnn8         | **91.1**    | **98.5%**     | **98.8%**     | **63.4**   | 90.2%        | **97.1%**    |

### Test
If you have our model and official test-set, you need to set official test-set pictures to '/AmurTigerReID/database/test/'

You only change the test.py as follow:

```
net = tiger_cnn1(classes=107)   # please change the target model name
net.load_state_dict(torch.load('./model/tiger_cnn1/model.ckpt')['net_state_dict'])  # please change the target model path
```

And then, you can run the test.py to get result.json.
