## DataSet
### Recent Updates
[2019.8.1]

* train.txt: official train-set, training model.

* gallery.txt and probe.txt: only validate the model, and it doesn't train our model.

* test.txt: official test-set, non-labeled, and it only outputs the results by the training model.

### train.txt  
the image path is relative path.

| image_path | class_name | left(0)/right(1)|
| :--------: | :--------: | :-------------: |
| train/0/000384.jpg | 0 | 0 |
| train/0/000584.jpg | 0 | 0 |
| ... | ... | ... |

### gallery.txt
the image path is relative path.

| image_path | class_name |
| :--------: | :--------: |
| train/0/000384.jpg | 0 |
| train/0/000584.jpg | 0 |
| ... | ... |

### probe.txt
the image path is relative path.

| image_path | class_name |
| :--------: | :--------: |
| val/0/0002.jpg | 107 |
| val/0/0003.jpg | 107 |
| ... | ... |


### test.txt
the image path is relative path.

| image_path |
| :--------: |
| test/002107.jpg |
| test/001133.jpg |
| ... |
