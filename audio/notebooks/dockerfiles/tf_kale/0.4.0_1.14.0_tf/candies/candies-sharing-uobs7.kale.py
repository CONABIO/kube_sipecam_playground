import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def sack(CANDIES: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/kale-base-example/.candies_sharing.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    import random

    def get_handful(left):
        if left == 0:
            print("There are no candies left! I want to cry :(")
            return 0
        c = random.randint(1, left)
        print("I got %s candies!" % c)
        return c

    print("Let's put in a bag %s candies and have three kids get a handful of them each" % CANDIES)


def kid1(CANDIES: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/kale-base-example/.candies_sharing.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    import random

    def get_handful(left):
        if left == 0:
            print("There are no candies left! I want to cry :(")
            return 0
        c = random.randint(1, left)
        print("I got %s candies!" % c)
        return c

    # kid1 gets a handful, without looking in the bad!
    kid1 = get_handful(CANDIES)

    # -----------------------DATA SAVING START---------------------------------
    if "kid1" in locals():
        _kale_resource_save(kid1, os.path.join(_kale_data_directory, "kid1"))
    else:
        print("_kale_resource_save: `kid1` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def kid2(CANDIES: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/kale-base-example/.candies_sharing.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "kid1" not in _kale_directory_file_names:
        raise ValueError("kid1" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "kid1"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "kid1" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    kid1 = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import random

    def get_handful(left):
        if left == 0:
            print("There are no candies left! I want to cry :(")
            return 0
        c = random.randint(1, left)
        print("I got %s candies!" % c)
        return c

    kid2 = get_handful(CANDIES - kid1)

    # -----------------------DATA SAVING START---------------------------------
    if "kid2" in locals():
        _kale_resource_save(kid2, os.path.join(_kale_data_directory, "kid2"))
    else:
        print("_kale_resource_save: `kid2` not found.")
    if "kid1" in locals():
        _kale_resource_save(kid1, os.path.join(_kale_data_directory, "kid1"))
    else:
        print("_kale_resource_save: `kid1` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def kid3(CANDIES: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/kale-base-example/.candies_sharing.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "kid1" not in _kale_directory_file_names:
        raise ValueError("kid1" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "kid1"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "kid1" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    kid1 = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "kid2" not in _kale_directory_file_names:
        raise ValueError("kid2" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "kid2"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "kid2" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    kid2 = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import random

    def get_handful(left):
        if left == 0:
            print("There are no candies left! I want to cry :(")
            return 0
        c = random.randint(1, left)
        print("I got %s candies!" % c)
        return c

    kid3 = get_handful(CANDIES - kid1 - kid2)


sack_op = comp.func_to_container_op(
    sack, base_image='sipecam/audio-kale:0.4.0_2.1.0')


kid1_op = comp.func_to_container_op(
    kid1, base_image='sipecam/audio-kale:0.4.0_2.1.0')


kid2_op = comp.func_to_container_op(
    kid2, base_image='sipecam/audio-kale:0.4.0_2.1.0')


kid3_op = comp.func_to_container_op(
    kid3, base_image='sipecam/audio-kale:0.4.0_2.1.0')


@dsl.pipeline(
    name='candies-sharing-uobs7',
    description='Share some candies between three lovely kids'
)
def auto_generated_pipeline(CANDIES='20', vol_shared_volume='efs'):
    pvolumes_dict = OrderedDict()

    annotations = {}

    volume = dsl.PipelineVolume(pvc=vol_shared_volume)

    pvolumes_dict['/shared_volume/'] = volume

    sack_task = sack_op(CANDIES, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    sack_task.container.working_dir = "/shared_volume/notebooks/kale-base-example"
    sack_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    kid1_task = kid1_op(CANDIES, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after(sack_task)
    kid1_task.container.working_dir = "/shared_volume/notebooks/kale-base-example"
    kid1_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    kid2_task = kid2_op(CANDIES, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after(kid1_task)
    kid2_task.container.working_dir = "/shared_volume/notebooks/kale-base-example"
    kid2_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    kid3_task = kid3_op(CANDIES, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after(kid2_task)
    kid3_task.container.working_dir = "/shared_volume/notebooks/kale-base-example"
    kid3_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('')

    # Submit a pipeline run
    run_name = 'candies-sharing-uobs7_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
