Install properly torch version according to cuda driver version

https://github.com/pytorch/pytorch/issues/4546

Check if torch is using the gpu

https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu

https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870

Monitor:

watch -n 3 nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv

Cifar10 example:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

https://discuss.pytorch.org/t/how-to-convert-layer-list-in-nn-module-to-cpu-or-gpu/36223/2

Cifar10 kale example:

https://github.com/kubeflow-kale/examples/blob/master/pytorch-classification/cifar10_classification.ipynb


Last notebook used [deployments/audio/kale-jupyterlab-kubeflow_0.4.0_1.14.0_tf_cpu](https://github.com/CONABIO/kube_sipecam/blob/master/deployments/audio/kale-jupyterlab-kubeflow_0.4.0_1.14.0_tf_cpu.yaml) **observe is cpu** for kale+kubeflow functionality according to [kube_sipecam/issues/8](https://github.com/CONABIO/kube_sipecam/issues/8). [audio/tf_kale/0.4.0_1.14.0_tf/Dockerfile](https://github.com/CONABIO/kube_sipecam/blob/master/dockerfiles/audio/tf_kale/0.4.0_1.14.0_tf/Dockerfile) is just used to transform notebook to kubeflow pipeline (as it doesn't requires a node with gpu) but the docker image that is used inside notebook (and also each step of pipeline) is: [dockerfiles/audio/tf_kale/0.4.0_2.1.0/Dockerfile](https://github.com/CONABIO/kube_sipecam/blob/master/dockerfiles/audio/tf_kale/0.4.0_2.1.0/Dockerfile)

Check files in this dir:

```
cifar10_classification.ipynb
torch-example-cifar10-h5con.kale.py

```
