# pytorch-ms-df-analyzer

This a project that can anany computing graphs in pytorch from many aspects.

# How to install docker evnvironment

1. install cuda-toolkit from official documents:

```
from https://developer.nvidia.com/zh-cn/cuda-downloads
```

2.  proxy setting:

```
need to set proxy of docker pull and apt-get
```


3.  install docker and kubernetes from offical tutorials: 

```
from https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/
from https://docs.docker.com/engine/install/ubuntu/
```

4. install nvidia-docker:

```
installation guide from https://github.com/NVIDIA/nvidia-docker
```

5. install kubeshare

```
https://github.com/NTHU-LSALAB/KubeShare
```

6. do necessary check

```
apiVersion: kubeshare.nthu/v1
kind: SharePod
metadata:
  name: sharepod1
  annotations:
    "kubeshare/gpu_request": "0.5" # required if allocating GPU
    "kubeshare/gpu_limit": "1.0" # required if allocating GPU
    "kubeshare/gpu_mem": "1073741824" # required if allocating GPU # 1Gi, in bytes
    #"kubeshare/sched_affinity": "red" # optional
    #"kubeshare/sched_anti-affinity": "green" # optional
    #"kubeshare/sched_exclusion": "blue" # optional
spec: # PodSpec
  containers:
  - name: cuda
    image: nvidia/cuda:11.0-base
    #command: ["nvidia-smi", "-L"]
    command: [ "/bin/bash", "-ce", "tail -f /dev/null" ]
    resources:
      limits:
        cpu: "1"
        memory: "500Mi"
```