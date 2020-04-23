see: [kubeflow-kale examples](https://github.com/kubeflow-kale/examples)

curl https://raw.githubusercontent.com/kubeflow-kale/examples/master/titanic-ml-dataset/titanic_dataset_ml.ipynb -o titanic_dataset_ml.ipynb

wget https://raw.githubusercontent.com/kubeflow-kale/examples/master/titanic-ml-dataset/data/test.csv

wget https://raw.githubusercontent.com/kubeflow-kale/examples/master/titanic-ml-dataset/data/train.csv

see: [jupyterlab-kubeflow-kale](https://github.com/kubeflow-kale/jupyterlab-kubeflow-kale)


```
sudo kale --nb titanic_dataset_ml.ipynb --experiment_name default --pipeline_name titanicml --debug

sudo kale --nb titanic_dataset_ml.ipynb --experiment_name default --pipeline_name titanicml --kfp_host <host>:<port> --upload_pipeline --debug

```

**old:**

wget https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv

mkdir data

mv titanic.csv data/
