### 分析过程
1. 数据来源：基于tensorBoard callback方法记录的Tensorflow模型训练时的迭代轨迹数据，json格式存放，包括时间戳、算子名、运行时间3类关键数据；每次试验跑2次，分别对应cpu和gpu；
2. 分析方式：基于json格式的轨迹数据交叉对比对CPU和GPU算子的平均运行时间；
3. 加速比统计：以CPU时间/GPU时间统计相同算子在不同设备上的平均运行时间差异。

### TensorFlow & TensorBoard分析API

1. 首先定义日志位置和callback函数，指定profile的迭代区间：

```
logs = "/tmp/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')
```

2. 基于keras API导入callback函数：

```
model.fit(ds_train,
          epochs=2,
          validation_data=ds_test,
          callbacks = [tboard_callback])
```

3. 解压log文件夹中的json文件，使用本项目的分析代码，即可得到算子的加速比。

### 预定义环境
基于conda的虚拟python环境：

pytorch可以兼容python最新版：

```
conda activate py38
```

TensorFlow只能用python旧版本：

```
conda activate py36
```


### TensorFlow official models的统计方法

需要改造官方模型仓库的https://github.com/Xuyuanjia2014/models代码，为其增加callback函数即可，official models中的tboard_callback添加位置，如下表所示（按照不同类别区分模型，模型本身已经自带相对稳定的benchmark数据集）：

| 模型名称 | 类别 | 修改位置 | 注意事项 |
| ------ | ------ | ------ | ------ |
|  MNIST <br /> ResNet <br/> EfficientNet   | official 图像分类  | [mnist_main.py](https://github.com/Xuyuanjia2014/models/blob/master/official/vision/image_classification/mnist_main.py)  122行；<br /> [classifier_trainer.py](https://github.com/Xuyuanjia2014/models/blob/master/official/vision/image_classification/classifier_trainer.py) 364行；  | 3个模型对应2个benchmark脚本  |
|  RetinaNet <br /> Mask R-CNN <br/> ShapeMask   | official 物体识别分类  | [main.py](https://github.com/Xuyuanjia2014/models/blob/master/official/vision/detection/main.py) 324行；  | 3个模型对应1个benchmark脚本  |
|  ALBERT (A Lite BERT)   | official 自然语言处理  | [run_classifier.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/bert/run_classifier.py) 242行； <br/> [run_squad.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/albert/run_squad.py) | 1个模型区分训练阶段（pre-train和fine tune）54行，有不同数据集  |
|  BERT   | official 自然语言处理  | [run_classifier.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/bert/run_classifier.py) 242行； <br/> [run_squad.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/bert/run_squad.py) 57行 | 1个模型区分训练阶段（pre-train和fine tune），有不同数据集，需要数据预处理  |
|  NHNet   | official 自然语言处理  | [trainer.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/transformer/transformer_main.py) 149行 | 需要pip安装其他依赖，需要数据预处理  |
|  Transformer  | official 自然语言处理  | [transformer_main.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/transformer/transformer_main.py) 347行 | 需要pip安装其他依赖，需要数据预处理  |
|  XLNet  | official 自然语言处理  | [transformer_main.py](https://github.com/Xuyuanjia2014/models/blob/master/official/nlp/xlnet/run_classifier.py) 158行 | 与bert类似，但缺少命令，不过目测与bert操作的方式和参数一致  |
|  NCF  | 推荐系统  | [ncf_keras_main.py](https://github.com/Xuyuanjia2014/models/blob/master/official/recommendation/ncf_keras_main.py) 321行 | 需要预下载数据  |

### 问题解决与注意事项

1. MNIST等数据集无法下载的问题

运行Proxy(切换文件夹中proxy的顺序，然后停止ctrl+c再启动，可以使用不同proxy，默认使用第一个)

```
/root/clash/clash -d .

```
新起命令行

```
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7891

curl www.google.com
```
如果有返回值就正常


2. ubuntu下的下载工具

```
cd /root/disk/utorrent-server-alpha-v3_3
./utserver
```

新启窗口：

```
lsof -i:8080
```

打开本地浏览器，输入网址(可以下载utorrent种子,用户名admin，密码为空)：
```
http://ip:8080/gui/
```

3. CentOS 7 64bit下的下载工具

需要下载旧版本的openssl： http://rpmfind.net/linux/rpm2html/search.php?query=openssl098e：

```
rpm -ivh *.rpm
```

按照https://ylface.com/server/376要求安装旧版本


4. linux间通过scp传输数据集

```
scp root@39.107.241.0:/root/disk/tensorflow-datasource/downloads/manual/ILSVRC2012_img_train.tar /root/tensorflow-datasource
```

5. 数据预处理时的问题(对imagenet 2012等需要手动下载并处理的datasets)

需要使用脚本：[python -m tensorflow_datasets.scripts.download_and_prepare \ 
 --datasets=imagenet2012](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md)

6. models、datasets和tf版本的对应关系

202008时间，一般需要拉取TensorFlow2.3以上版本的支持，否则会出现编译选项等多种异常。





