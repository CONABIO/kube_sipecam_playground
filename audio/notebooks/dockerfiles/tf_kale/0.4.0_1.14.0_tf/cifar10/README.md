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
