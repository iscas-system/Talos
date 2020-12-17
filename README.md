# Talos

Talos is a dataflow analysis and scheduling tool for deep learning applications.

# operator speedup analysis for pytorch

[click here to see pytorch operator chrome tracing](pytorch-analyzer)

# operator speedup analysis for TensorFlow

[click here to see tensorflow operator chrome tracing](tensorflow-analyzer)

# operator speedup aware scheduling

[click here to see operator speedup awareness scheudling](java-operator-scheduler)

# road map

1. Select a proper deep learning framework to that can easily control operation location.
2. Modify an existing framework to support operator speedup ratio awareness scheduling.
3. integrate existing data or model parallel mechanisms to make the scheduling practical.
4. optimize existing device placement granularity and strategy. 

# problems 

## code completion

```
ctrl+shift+p: open user settings

copy and change path:

{
    "python.autoComplete.addBrackets": true,
    "python.autoComplete.extraPaths": [
        // "/root/anaconda3/envs/xyj_pytorch",
        // "/root/anaconda3/envs/xyj_pytorch/lib/python38.zip",
        // "/root/anaconda3/envs/xyj_pytorch/lib/python3.8",
        // "/root/anaconda3/envs/xyj_pytorch/lib/python3.8/lib-dynload",
        // "/root/anaconda3/envs/xyj_pytorch/lib/python3.8/site-packages",
        "/root/anaconda3/envs/d2l",
        "/root/anaconda3/envs/d2l/lib/python38.zip",
        "/root/anaconda3/envs/d2l/lib/python3.8",
        "/root/anaconda3/envs/d2l/lib/python3.8/lib-dynload",
        "/root/anaconda3/envs/d2l/lib/python3.8/site-packages"
    ]
}
```

## d2l has some dependencies:

```
export PYTHONPATH=/root/d2l-en/mxnet 或 pip install d2l==0.13.2 -f https://d2l.ai/whl.html
pip install ipython
pip install pandas
pip install ipykernel
```

## comments on dive to deep learning 

1. 机器学习的过程：优化和泛化
2. 计算图可以标记常量，避免全部微分，也能够控制微分分支
3. 