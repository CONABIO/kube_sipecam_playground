import kfp.dsl as dsl
import json
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def variables():
    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    block1 = '''
    import os
    import sys
    import numpy as np
    import pandas as pd
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from conabio_ml.pipeline import Pipeline
    from conabio_ml.datasets.dataset import Dataset
    from conabio_ml.utils.report_params import languages
    import conabio_ml.utils.images_utils as images_utils
    from conabio_ml.trainer.model import Model as CBModel
    from conabio_ml.trainer.images.model import ClassificationModel
    from conabio_ml.trainer.images.predictor_config import TFPredictorConfig
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, ImageDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator,ImageClassificationMetrics
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasTrainerConfig, TFKerasTrainer, CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK, MIRROREDSTRATEGY
    from conabio_ml.preprocessing import TFKerasPreprocessor
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasModel
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator as Evaluator
    from conabio_ml.evaluator.images.evaluator import ImageClassificationMetrics as Metrics

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.preprocessing import image
    import tensorflow_hub as tfhub
    '''

    block2 = '''
    IMG_HEIGHT = 299
    IMG_WIDTH = 299
    BATCH_SIZE = 32
    TRAIN_PERC = 0.8
    VAL_PERC = 0.1
    exp_name = 'exp-2'
    NUM_EPOCHS = 50
    eval_type = 'weighted'

    exp_files_path = "/shared_volume/files"
    crops_dataset_path = os.path.join(exp_files_path, "crops_dataset.csv")
    images_dir = os.path.join(exp_files_path, 'images')
    snmb_mapping_csv_path =  os.path.join(exp_files_path, "mappings_1.csv")
    crops_images_path = os.path.join(exp_files_path, "images_crops")
    snmb_json_path = os.path.join(exp_files_path, "snmb_2020_detection.json")

    results_path = os.path.join(exp_files_path, 'results', exp_name)
    train_dir = os.path.join(results_path, f'train_{NUM_EPOCHS:02}_epochs')
    classif_on_crops_dataset_path = os.path.join(train_dir, "classifs_on_crops.csv")
    results_eval_path = os.path.join(train_dir, f"results_eval_{eval_type}.json")

    os.makedirs(crops_images_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.save(BATCH_SIZE, "BATCH_SIZE")
    _kale_marshal_utils.save(IMG_HEIGHT, "IMG_HEIGHT")
    _kale_marshal_utils.save(IMG_WIDTH, "IMG_WIDTH")
    _kale_marshal_utils.save(NUM_EPOCHS, "NUM_EPOCHS")
    _kale_marshal_utils.save(TRAIN_PERC, "TRAIN_PERC")
    _kale_marshal_utils.save(VAL_PERC, "VAL_PERC")
    _kale_marshal_utils.save(classif_on_crops_dataset_path, "classif_on_crops_dataset_path")
    _kale_marshal_utils.save(crops_dataset_path, "crops_dataset_path")
    _kale_marshal_utils.save(crops_images_path, "crops_images_path")
    _kale_marshal_utils.save(eval_type, "eval_type")
    _kale_marshal_utils.save(images_dir, "images_dir")
    _kale_marshal_utils.save(results_eval_path, "results_eval_path")
    _kale_marshal_utils.save(snmb_json_path, "snmb_json_path")
    _kale_marshal_utils.save(snmb_mapping_csv_path, "snmb_mapping_csv_path")
    _kale_marshal_utils.save(train_dir, "train_dir")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (
        block1,
        block2,
        data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/variables.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('variables')

    _kale_mlmd_utils.call("mark_execution_complete")


def dataset():
    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.set_kale_directory_file_names()
    TRAIN_PERC = _kale_marshal_utils.load("TRAIN_PERC")
    VAL_PERC = _kale_marshal_utils.load("VAL_PERC")
    crops_dataset_path = _kale_marshal_utils.load("crops_dataset_path")
    crops_images_path = _kale_marshal_utils.load("crops_images_path")
    images_dir = _kale_marshal_utils.load("images_dir")
    snmb_json_path = _kale_marshal_utils.load("snmb_json_path")
    snmb_mapping_csv_path = _kale_marshal_utils.load("snmb_mapping_csv_path")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import os
    import sys
    import numpy as np
    import pandas as pd
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from conabio_ml.pipeline import Pipeline
    from conabio_ml.datasets.dataset import Dataset
    from conabio_ml.utils.report_params import languages
    import conabio_ml.utils.images_utils as images_utils
    from conabio_ml.trainer.model import Model as CBModel
    from conabio_ml.trainer.images.model import ClassificationModel
    from conabio_ml.trainer.images.predictor_config import TFPredictorConfig
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, ImageDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator,ImageClassificationMetrics
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasTrainerConfig, TFKerasTrainer, CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK, MIRROREDSTRATEGY
    from conabio_ml.preprocessing import TFKerasPreprocessor
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasModel
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator as Evaluator
    from conabio_ml.evaluator.images.evaluator import ImageClassificationMetrics as Metrics

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.preprocessing import image
    import tensorflow_hub as tfhub
    '''

    block2 = '''
    if os.path.isfile(crops_dataset_path):
        crops_dataset = ImageDataset.from_csv(source_path=crops_dataset_path,
                                               images_dir=crops_images_path)
    else:
        dataset = ConabioImageDataset.from_json(source_path=snmb_json_path, 
                                               images_dir=images_dir,
                                               mapping_classes=snmb_mapping_csv_path,
                                               categories=["HRTL", "COSCM", "COTB", "FrOSCSM", "CSCB", "HPTM", "CSCL", "FrOSCM", "IOTM", "CSFB", "FrOTL", "FFS", "GFS", "HFS", "FFSA", "OFS", "FrHTL", "FrOSCB", "COSCB", "CTB", "IOTB", "IOSFM", "HPTB", "FrOAM", "CSFM", "PSQB", "FrOTB", "CTL"],
                                               exclude_categories=["Equus asinus", "Equus caballus", "Capra hircus", "Capra ibex", "Capra nubiana", "Vulpes vulpes", "Felis silvestris", "Oryctolagus cuniculus"])
        crops_dataset = dataset.create_classification_dataset_from_bboxes_crops(dest_path=crops_images_path, 
                                                                                include_id=True,
                                                                                inherit_fields=["image_id", "location"])
        crops_dataset.split(train_perc=TRAIN_PERC,
                            val_perc=VAL_PERC,
                             test_perc=1 - (TRAIN_PERC+VAL_PERC),
                             group_by_field="location")
        crops_dataset.to_csv(dest_path=crops_dataset_path,
                              columns=["item", "label"],
                              include_relative_path_in_items=False)
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.save(crops_dataset, "crops_dataset")
    _kale_marshal_utils.save(crops_dataset_path, "crops_dataset_path")
    _kale_marshal_utils.save(crops_images_path, "crops_images_path")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (data_loading_block,
              block1,
              block2,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/dataset.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('dataset')

    _kale_mlmd_utils.call("mark_execution_complete")


def model():
    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    block1 = '''
    import os
    import sys
    import numpy as np
    import pandas as pd
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from conabio_ml.pipeline import Pipeline
    from conabio_ml.datasets.dataset import Dataset
    from conabio_ml.utils.report_params import languages
    import conabio_ml.utils.images_utils as images_utils
    from conabio_ml.trainer.model import Model as CBModel
    from conabio_ml.trainer.images.model import ClassificationModel
    from conabio_ml.trainer.images.predictor_config import TFPredictorConfig
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, ImageDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator,ImageClassificationMetrics
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasTrainerConfig, TFKerasTrainer, CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK, MIRROREDSTRATEGY
    from conabio_ml.preprocessing import TFKerasPreprocessor
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasModel
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator as Evaluator
    from conabio_ml.evaluator.images.evaluator import ImageClassificationMetrics as Metrics

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.preprocessing import image
    import tensorflow_hub as tfhub
    '''

    block2 = '''
    class InceptionModel(TFKerasModel):
        INCEPTION_MODEL = "InceptionModel"
        IMG_HEIGHT = 299
        IMG_WIDTH = 299
        _report_data = {
            INCEPTION_MODEL: {
                "model_name": {
                    languages.EN: "Inception model",
                    languages.ES: "Modelo Inception"
                },
                "model_type": {
                    languages.EN: f"Convolutional neural network and a pure Inception variant without "
                    f"residual connections. "
                    f"The pretrained network can classify images into 1000 object categories.",
                    languages.ES: f"Red neuronal convolucional y una variante de Inception sin "
                    f"conexiones residuales. "
                    f"La red preentrenada puede clasificar im\xe1genes en 1000 categor\xedas de objetos."
                },
                "input_data": {
                    languages.EN: "The network has an image input size of 299-by-299.",
                    languages.ES: "La red tiene un tama\xf1o de entrada de imagen de 299 por 299."
                },
                "output_data": {
                    languages.EN: f"Output layer is softmax, which means it has "
                    f"predefined number of neurons, each one is defined for one specific class.",
                    languages.ES: f"La capa de salida es softmax, lo que significa que tiene un n\xfamero"
                    f" predefinido de neuronas, cada una est\xe1 definida para una clase espec\xedfica."
                }
            }
        }
        report_data = {**ClassificationModel.report_data, **_report_data}

        @classmethod
        def create_model(cls: CBModel.ModelType, layer_config: dict, num_categories: int) -> CBModel.ModelType:
            model = tf.keras.Sequential([
                tfhub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4",
                                 trainable=True, arguments=dict(batch_norm_momentum=0.997)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(num_categories,
                                      activation="softmax",
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            ])

            return model

        @classmethod
        def load_model(cls: CBModel.ModelType, model_path: str) -> CBModel.ModelType:
            instance = InceptionModel({
                    "InceptionModel": {
                        "layers": {}
                    }
                })
            instance.model = load_model(model_path)
            return instance

        def predict(self: CBModel.ModelType, 
                    dataset: ImageDataset.DatasetType, 
                    execution_config: TFPredictorConfig.TFPredictorConfigType, 
                    prediction_config: dict) -> Dataset.DatasetType:
            results = {
                "item": [],
                "label": [],
                "score": []
            }
                    
            labelmap = dataset.get_labelmap()
            partition = prediction_config.get("dataset_partition", "test")
            max_classifs = prediction_config.get('max_classifs', None)
            batch_size = execution_config.batch_size

            df_test = dataset.get_rows(partition)
            items = df_test["item"].unique()
            items_batches = images_utils.get_batches(items, batch_size) 
            for items_batch in items_batches:
                images_dict = images_utils.load_image_into_numpy_array_batch_keras(
                    items_batch, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))
                images = [image_np for image_np in images_dict.values() if image_np is not None]
                items = [item for item, image_np in images_dict.items() if image_np is not None]
                preds = self.model.predict(np.vstack(images), batch_size=batch_size)
                for j in range(len(preds)):
                    probs = preds[j, 0:]
                    sorted_inds = [y[0] for y in sorted(enumerate(-probs), key=lambda x:x[1])]
                    if max_classifs is None:
                        max_clasif_var = len(sorted_inds)
                    else:
                        max_clasif_var = min(max_classifs, len(sorted_inds))
                    for k in range(max_clasif_var):
                        ind = sorted_inds[k]
                        results["item"].append(items[j])
                        results["label"].append(labelmap[ind])
                        results["score"].append(probs[ind])
                        
            prediction_data = {
                "prediction_type": "classification",
                "thresholds": {
                    "max_classifs": max_classifs
                }
            }
            data = pd.DataFrame(results)
            prediction_dataset = ImagePredictionDataset(data, info={}, prediction_data=prediction_data)
            return prediction_dataset
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.save(InceptionModel, "InceptionModel")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (
        block1,
        block2,
        data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/model.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('model')

    _kale_mlmd_utils.call("mark_execution_complete")


def train():
    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.set_kale_directory_file_names()
    BATCH_SIZE = _kale_marshal_utils.load("BATCH_SIZE")
    IMG_HEIGHT = _kale_marshal_utils.load("IMG_HEIGHT")
    IMG_WIDTH = _kale_marshal_utils.load("IMG_WIDTH")
    InceptionModel = _kale_marshal_utils.load("InceptionModel")
    NUM_EPOCHS = _kale_marshal_utils.load("NUM_EPOCHS")
    crops_dataset = _kale_marshal_utils.load("crops_dataset")
    train_dir = _kale_marshal_utils.load("train_dir")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import os
    import sys
    import numpy as np
    import pandas as pd
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from conabio_ml.pipeline import Pipeline
    from conabio_ml.datasets.dataset import Dataset
    from conabio_ml.utils.report_params import languages
    import conabio_ml.utils.images_utils as images_utils
    from conabio_ml.trainer.model import Model as CBModel
    from conabio_ml.trainer.images.model import ClassificationModel
    from conabio_ml.trainer.images.predictor_config import TFPredictorConfig
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, ImageDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator,ImageClassificationMetrics
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasTrainerConfig, TFKerasTrainer, CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK, MIRROREDSTRATEGY
    from conabio_ml.preprocessing import TFKerasPreprocessor
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasModel
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator as Evaluator
    from conabio_ml.evaluator.images.evaluator import ImageClassificationMetrics as Metrics

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.preprocessing import image
    import tensorflow_hub as tfhub
    '''

    block2 = '''
    checkpoints_dir = os.path.abspath(os.path.join(train_dir, "checkpoints"))
    if not os.path.isfile(os.path.abspath(os.path.join(checkpoints_dir, "saved_model.pb"))):
        os.makedirs(checkpoints_dir, exist_ok=True)
        ds_gen = TFKerasPreprocessor.as_image_generator(dataset=crops_dataset,
                                                          preproc_args={
                                                            'use_partitions': True,
                                                            'preproc_opts': {
                                                                "rescale": 1./255,
                                                                'horizontal_flip': True,
                                                                'rotation_range': 20,
                                                                "zoom_range": 0.15,
                                                                "width_shift_range": 0.1,
                                                                "height_shift_range": 0.1,
                                                                "shear_range": 0.15
                                                            },
                                                            'dataset_handling': {
                                                                "target_size": (IMG_HEIGHT, IMG_WIDTH),
                                                                'batch_size': BATCH_SIZE
                                                            }
                                                        })
        exec_config = TFKerasTrainerConfig.create(config={
                                                     "callbacks": {
                                                         CHECKPOINT_CALLBACK: {
                                                             "filepath": checkpoints_dir,
                                                             "save_best_only": True
                                                         },
                                                         TENSORBOARD_CALLBACK: {
                                                             "log_dir": os.path.join(train_dir)
                                                         }
                                                     },
                                                     'strategy': {
                                                         MIRROREDSTRATEGY: {
                                                             "devices": []
                                                         }
                                                     }
                                                 })
        model = InceptionModel.create(model_config={
                                            "InceptionModel": {
                                                "layers": {}
                                            }
                                        })
        TFKerasTrainer.train(dataset=ds_gen,
                            model=model,
                            execution_config=exec_config,
                            train_config={
                                'InceptionModel': {
                                    'representation': 'image_generator',
                                    'optimizer': {
                                        'adam': {
                                            'learning_rate': {
                                                'constant': {
                                                    'learning_rate': 0.001,
                                                }
                                            }
                                        }
                                    },
                                    'loss': {
                                        'categorical_crossentropy': {}
                                    },
                                    'epochs': NUM_EPOCHS,
                                    "metrics": ["accuracy"]
                                }
                            })
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.save(BATCH_SIZE, "BATCH_SIZE")
    _kale_marshal_utils.save(InceptionModel, "InceptionModel")
    _kale_marshal_utils.save(checkpoints_dir, "checkpoints_dir")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (data_loading_block,
              block1,
              block2,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/train.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('train')

    _kale_mlmd_utils.call("mark_execution_complete")


def predict():
    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.set_kale_directory_file_names()
    BATCH_SIZE = _kale_marshal_utils.load("BATCH_SIZE")
    InceptionModel = _kale_marshal_utils.load("InceptionModel")
    checkpoints_dir = _kale_marshal_utils.load("checkpoints_dir")
    classif_on_crops_dataset_path = _kale_marshal_utils.load("classif_on_crops_dataset_path")
    crops_dataset_path = _kale_marshal_utils.load("crops_dataset_path")
    crops_images_path = _kale_marshal_utils.load("crops_images_path")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import os
    import sys
    import numpy as np
    import pandas as pd
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from conabio_ml.pipeline import Pipeline
    from conabio_ml.datasets.dataset import Dataset
    from conabio_ml.utils.report_params import languages
    import conabio_ml.utils.images_utils as images_utils
    from conabio_ml.trainer.model import Model as CBModel
    from conabio_ml.trainer.images.model import ClassificationModel
    from conabio_ml.trainer.images.predictor_config import TFPredictorConfig
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, ImageDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator,ImageClassificationMetrics
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasTrainerConfig, TFKerasTrainer, CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK, MIRROREDSTRATEGY
    from conabio_ml.preprocessing import TFKerasPreprocessor
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasModel
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator as Evaluator
    from conabio_ml.evaluator.images.evaluator import ImageClassificationMetrics as Metrics

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.preprocessing import image
    import tensorflow_hub as tfhub
    '''

    block2 = '''
    model = InceptionModel.load_model(model_path=checkpoints_dir)
    pred_config = TFPredictorConfig.create(batch_size=BATCH_SIZE,
                                          num_preprocessing_threads=8)
    dataset = ImageDataset.from_csv(source_path=crops_dataset_path,
                                    images_dir=crops_images_path)
    pred_dataset = model.predict(dataset=dataset,
                                 execution_config=pred_config,
                                 prediction_config={
                                    "dataset_partition": Partitions.TEST,
                                    "max_classifs": 1
                                 })
    pred_dataset.to_csv(dest_path=classif_on_crops_dataset_path)
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.save(classif_on_crops_dataset_path, "classif_on_crops_dataset_path")
    _kale_marshal_utils.save(crops_dataset_path, "crops_dataset_path")
    _kale_marshal_utils.save(crops_images_path, "crops_images_path")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (data_loading_block,
              block1,
              block2,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/predict.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('predict')

    _kale_mlmd_utils.call("mark_execution_complete")


def eval():
    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/shared_volume/third_tests/.test_train_gpu.ipynb.kale.marshal.dir")
    _kale_marshal_utils.set_kale_directory_file_names()
    classif_on_crops_dataset_path = _kale_marshal_utils.load("classif_on_crops_dataset_path")
    crops_dataset_path = _kale_marshal_utils.load("crops_dataset_path")
    crops_images_path = _kale_marshal_utils.load("crops_images_path")
    eval_type = _kale_marshal_utils.load("eval_type")
    results_eval_path = _kale_marshal_utils.load("results_eval_path")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import os
    import sys
    import numpy as np
    import pandas as pd
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from conabio_ml.pipeline import Pipeline
    from conabio_ml.datasets.dataset import Dataset
    from conabio_ml.utils.report_params import languages
    import conabio_ml.utils.images_utils as images_utils
    from conabio_ml.trainer.model import Model as CBModel
    from conabio_ml.trainer.images.model import ClassificationModel
    from conabio_ml.trainer.images.predictor_config import TFPredictorConfig
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, ImageDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator,ImageClassificationMetrics
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasTrainerConfig, TFKerasTrainer, CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK, MIRROREDSTRATEGY
    from conabio_ml.preprocessing import TFKerasPreprocessor
    from conabio_ml.trainer.backends.tfkeras_bcknd import TFKerasModel
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator as Evaluator
    from conabio_ml.evaluator.images.evaluator import ImageClassificationMetrics as Metrics

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.preprocessing import image
    import tensorflow_hub as tfhub
    '''

    block2 = '''
    dataset_pred = ImagePredictionDataset.from_csv(source_path=classif_on_crops_dataset_path,
                                                  images_dir=crops_images_path)
    dataset_true = ImageDataset.from_csv(source_path=crops_dataset_path,
                                         images_dir=crops_images_path)
    metrics = Evaluator.eval(dataset_true=dataset_true,
                             dataset_pred=dataset_pred,
                             eval_config={
                                "metrics_set": {
                                    Metrics.Sets.MULTICLASS: {
                                        "average": eval_type
                                    }
                                },
                                "dataset_partition": Partitions.TEST
                             })
    metrics.store_eval_metrics(results_eval_path)
    metrics.result_plots()
    '''

    block3 = '''
    from collections import defaultdict
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from conabio_ml.datasets.images.datasets import ConabioImageDataset, ImagePredictionDataset, Partitions
    from conabio_ml.evaluator.images.evaluator import ImageClassificationEvaluator, ImageClassificationMetrics

    crops_ds = ConabioImageDataset.from_csv(source_path=crops_dataset_path,
                                             images_dir=crops_images_path)
    classifs_ds = ImagePredictionDataset.from_csv(source_path=classif_on_crops_dataset_path,
                                                  images_dir=crops_images_path)

    eval_type = "MULTICLASS"
    average_type = 'weighted'
    zero_division = 0.

    classifs_df = classifs_ds.as_dataframe(only_highest_score=True, sort_by="item")
    labels = list(set(crops_ds.get_rows(Partitions.TEST)["label"]).union(set(classifs_df["label"])))

    if eval_type == "MULTILABEL":
        dataset_true = crops_ds.create_multilabel_classification_dataset_from_crops(
            images_dir="",
            extension_to_append="JPG",
            inherit_partitions=True,
            inherit_fields=["location"])
    else:
        dataset_true = crops_ds

    results_dict = defaultdict(list)
    for thres in range(0, 100, 10):
        if eval_type == "MULTILABEL":
            dataset_pred = classifs_ds.create_multiclass_dataset_from_classifs_on_crops(thres/100)
            eval_func = Evaluator.eval
        else:
            df_pred = classifs_ds.as_dataframe(only_highest_score=True, sort_by="item")
            df_pred.loc[df_pred["score"] < thres/100, "label"] = "empty"
            classifs_ds.data = df_pred
            dataset_pred = classifs_ds
            eval_func = ImageClassificationEvaluator.eval

        if len(dataset_pred.data.loc[dataset_pred.data["label"] == "empty"]) > 0:
            _labels = labels + ["empty"]
        else:
            _labels = labels
        
        metrics = eval_func(dataset_true=dataset_true,
                            dataset_pred=dataset_pred,
                            eval_config={
                                'metrics_set': {
                                    eval_type: {
                                        "average": average_type,
                                        "zero_division": zero_division
                                    }
                                },
                                'dataset_partition': Partitions.TEST,
                                "labels": _labels
                            })
        _metrics = ["precision", "recall", "f1_score"]
        if eval_type == "MULTICLASS":
            _metrics.append("accuracy")
        for metric in _metrics:
            results_dict["threshold"].append(thres/100)
            results_dict["value"].append(metrics.results[eval_type]["one_class"][metric])
            results_dict["type"].append(metric)
    df = pd.DataFrame(results_dict)
    ax = sns.lineplot(data=df, x="threshold", y="value", hue="type")
    ax.set(xlabel='Threshold', ylabel='Metric value')
    ax.grid()
    plt.show()
    '''

    block4 = '''
    
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (data_loading_block,
              block1,
              block2,
              block3,
              block4,
              )
    html_artifact = _kale_run_code(blocks)
    with open("/eval.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('eval')

    _kale_mlmd_utils.call("mark_execution_complete")


variables_op = comp.func_to_container_op(
    variables, base_image='sipecam/ecoinf-kale-gpu:0.5.0')


dataset_op = comp.func_to_container_op(
    dataset, base_image='sipecam/ecoinf-kale-gpu:0.5.0')


model_op = comp.func_to_container_op(
    model, base_image='sipecam/ecoinf-kale-gpu:0.5.0')


train_op = comp.func_to_container_op(
    train, base_image='sipecam/ecoinf-kale-gpu:0.5.0')


predict_op = comp.func_to_container_op(
    predict, base_image='sipecam/ecoinf-kale-gpu:0.5.0')


eval_op = comp.func_to_container_op(
    eval, base_image='sipecam/ecoinf-kale-gpu:0.5.0')


@dsl.pipeline(
    name='test-train-gpu-mkkum',
    description='Entrenamiento usando GPU'
)
def auto_generated_pipeline(vol_shared_volume='hostpath-pvc'):
    pvolumes_dict = OrderedDict()
    volume_step_names = []
    volume_name_parameters = []

    annotations = {}

    volume = dsl.PipelineVolume(pvc=vol_shared_volume)

    pvolumes_dict['/shared_volume'] = volume

    volume_step_names.sort()
    volume_name_parameters.sort()

    variables_task = variables_op()\
        .add_pvolumes(pvolumes_dict)\
        .after()
    variables_task.container.working_dir = "/shared_volume/third_tests"
    variables_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'variables': '/variables.html'})
    variables_task.output_artifact_paths.update(output_artifacts)
    variables_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = variables_task.dependent_names + volume_step_names
    variables_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        variables_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    dataset_task = dataset_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(variables_task)
    dataset_task.container.working_dir = "/shared_volume/third_tests"
    dataset_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'dataset': '/dataset.html'})
    dataset_task.output_artifact_paths.update(output_artifacts)
    dataset_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = dataset_task.dependent_names + volume_step_names
    dataset_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        dataset_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    model_task = model_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(dataset_task)
    model_task.container.working_dir = "/shared_volume/third_tests"
    model_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'model': '/model.html'})
    model_task.output_artifact_paths.update(output_artifacts)
    model_task.add_pod_label("pipelines.kubeflow.org/metadata_written", "true")
    dep_names = model_task.dependent_names + volume_step_names
    model_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        model_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    train_task = train_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(model_task)
    train_task.container.working_dir = "/shared_volume/third_tests"
    train_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'train': '/train.html'})
    train_task.output_artifact_paths.update(output_artifacts)
    train_task.add_pod_label("pipelines.kubeflow.org/metadata_written", "true")
    dep_names = train_task.dependent_names + volume_step_names
    train_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        train_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    predict_task = predict_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(train_task)
    predict_task.container.working_dir = "/shared_volume/third_tests"
    predict_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'predict': '/predict.html'})
    predict_task.output_artifact_paths.update(output_artifacts)
    predict_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = predict_task.dependent_names + volume_step_names
    predict_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        predict_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    eval_task = eval_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(predict_task)
    eval_task.container.working_dir = "/shared_volume/third_tests"
    eval_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'eval': '/eval.html'})
    eval_task.output_artifact_paths.update(output_artifacts)
    eval_task.add_pod_label("pipelines.kubeflow.org/metadata_written", "true")
    dep_names = eval_task.dependent_names + volume_step_names
    eval_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        eval_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('exp-3')

    # Submit a pipeline run
    from kale.utils.kfp_utils import generate_run_name
    run_name = generate_run_name('test-train-gpu-mkkum')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
