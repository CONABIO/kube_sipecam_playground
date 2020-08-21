import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def dataprocessing(TRAIN_STEPS: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/pytorch-classif/.cifar10_classification.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    import torch
    import torchvision
    import torchvision.transforms as transforms
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_data_folder = "./data"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=input_data_folder, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=input_data_folder, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # -----------------------DATA SAVING START---------------------------------
    if "testloader" in locals():
        _kale_resource_save(testloader, os.path.join(
            _kale_data_directory, "testloader"))
    else:
        print("_kale_resource_save: `testloader` not found.")
    if "trainloader" in locals():
        _kale_resource_save(trainloader, os.path.join(
            _kale_data_directory, "trainloader"))
    else:
        print("_kale_resource_save: `trainloader` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def train(TRAIN_STEPS: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/pytorch-classif/.cifar10_classification.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "trainloader" not in _kale_directory_file_names:
        raise ValueError("trainloader" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "trainloader"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "trainloader" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    trainloader = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import torch
    import torchvision
    import torchvision.transforms as transforms
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    import torch.optim as optim
    device = torch.device("cuda:0")
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0")
    for epoch in range(TRAIN_STEPS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # -----------------------DATA SAVING START---------------------------------
    if "net" in locals():
        _kale_resource_save(net, os.path.join(_kale_data_directory, "net"))
    else:
        print("_kale_resource_save: `net` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def testontest(TRAIN_STEPS: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/pytorch-classif/.cifar10_classification.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "testloader" not in _kale_directory_file_names:
        raise ValueError("testloader" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "testloader"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "testloader" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    testloader = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "net" not in _kale_directory_file_names:
        raise ValueError("net" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "net"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "net" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    net = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import torch
    import torchvision
    import torchvision.transforms as transforms
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    device = torch.device("cuda:0")
    dataiter = iter(testloader)
    n = dataiter.next()
    images, labels = n[0].to(device), n[1].to(device)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    # -----------------------DATA SAVING START---------------------------------
    if "testloader" in locals():
        _kale_resource_save(testloader, os.path.join(
            _kale_data_directory, "testloader"))
    else:
        print("_kale_resource_save: `testloader` not found.")
    if "net" in locals():
        _kale_resource_save(net, os.path.join(_kale_data_directory, "net"))
    else:
        print("_kale_resource_save: `net` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def testwhole(TRAIN_STEPS: int, vol_shared_volume: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/shared_volume/notebooks/pytorch-classif/.cifar10_classification.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "testloader" not in _kale_directory_file_names:
        raise ValueError("testloader" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "testloader"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "testloader" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    testloader = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "net" not in _kale_directory_file_names:
        raise ValueError("net" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "net"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "net" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    net = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import torch
    import torchvision
    import torchvision.transforms as transforms
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    device = torch.device("cuda:0")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    device = torch.device("cuda:0")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


dataprocessing_op = comp.func_to_container_op(
    dataprocessing, base_image='sipecam/audio-kale:0.4.0_2.1.0')


train_op = comp.func_to_container_op(
    train, base_image='sipecam/audio-kale:0.4.0_2.1.0')


testontest_op = comp.func_to_container_op(
    testontest, base_image='sipecam/audio-kale:0.4.0_2.1.0')


testwhole_op = comp.func_to_container_op(
    testwhole, base_image='sipecam/audio-kale:0.4.0_2.1.0')


@dsl.pipeline(
    name='torch-example-cifar10-h5con',
    description='Example of usage of torch and kale'
)
def auto_generated_pipeline(TRAIN_STEPS='2', vol_shared_volume='efs'):
    pvolumes_dict = OrderedDict()

    annotations = {}

    volume = dsl.PipelineVolume(pvc=vol_shared_volume)

    pvolumes_dict['/shared_volume/'] = volume

    dataprocessing_task = dataprocessing_op(TRAIN_STEPS, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    dataprocessing_task.container.working_dir = "/shared_volume/notebooks/pytorch-classif"
    dataprocessing_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    train_task = train_op(TRAIN_STEPS, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after(dataprocessing_task)
    train_task.container.working_dir = "/shared_volume/notebooks/pytorch-classif"
    train_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    testontest_task = testontest_op(TRAIN_STEPS, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after(train_task)
    testontest_task.container.working_dir = "/shared_volume/notebooks/pytorch-classif"
    testontest_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    testwhole_task = testwhole_op(TRAIN_STEPS, vol_shared_volume)\
        .add_pvolumes(pvolumes_dict)\
        .after(testontest_task)
    testwhole_task.container.working_dir = "/shared_volume/notebooks/pytorch-classif"
    testwhole_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('cifar10')

    # Submit a pipeline run
    run_name = 'torch-example-cifar10-h5con_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
