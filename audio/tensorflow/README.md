# README

Tutorials in: [tensorflow tutorials](https://www.tensorflow.org/tutorials/)

* Check tensorboard with tutorial [get_started](https://www.tensorflow.org/tensorboard/get_started) (if used in cmd line open port 6006, see [step5](https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/notebooks/step5.ipynb). Also see [TensorBoard: Visualizing Learning](https://github.com/tensorflow/tensorboard/blob/master/docs/r1/summaries.md)

* TFX [airflow workshop](https://www.tensorflow.org/tfx/tutorials/tfx/airflow_workshop). Github in [airflow workshop github](https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop). Important files: [taxi_pipeline_solution.py](https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/setup/dags/taxi_pipeline_solution.py) and [taxi_utils_solution.py](https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/setup/dags/taxi_utils_solution.py)

	* Link uses airflow. Also in that link refers to some components such as [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) or [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen). This components are part from tfx as they are imported as this [lines](https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/setup/dags/taxi_pipeline_solution.py#L33)

* **But I recommend see links and only use page of airflow workshop as a guide**

	* Link refers to [Tfx Data Validation](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic) as tools to inspect datasets. Also see [TensorFlow Data Validation](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic)
        * Link refers to [Tfx Transform](https://www.tensorflow.org/tfx/guide/transform) for feature engineering. Also see [Preprocessing data with Tensorflow Transform](https://www.tensorflow.org/tfx/tutorials/transform/census)
        * Link refers to [Tfx Trainer](https://www.tensorflow.org/tfx/guide/trainer) and [Estimators](https://www.tensorflow.org/guide/estimator). Also see [Estimators github docu](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/estimators.md)
        * Link refers to [Tfx Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) and [Tensorflow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma). Also see [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/tutorials/model_analysis/tfma_basic)
        * Link refers to [Tfx Pusher](https://www.tensorflow.org/tfx/guide/pusher) and [Saved model format](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model)
        * Link states "model ready for production" and refers to [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving), [Tensorflow Lite](https://www.tensorflow.org/lite), [Tensorflow.js](https://www.tensorflow.org/js)
