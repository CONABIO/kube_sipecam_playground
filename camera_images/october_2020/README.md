Labelled photos were used at the object level with a bounding box and the functional group to which the species of the individual present in the bounding box belongs. The functional groups were determined according to the ecological function that species play in an ecosystem, using criteria such as form of feeding, habitat, diet, size, etc.
From each photo, the regions of the annotations’ boxes were cropped and these images were taken to form a set of around 10,000 image-class pairs, which was partitioned in a proportion of 90%-10% for training and evaluation, respectively.
In the training stage, the Inception ResNet V2 model, previously trained with the Imagenet set, was re-trained for 50 epochs with the dataset described above, using the Keras library with the Tensorflow 2 interface.
At the end of the training, the precision, recall, score-F1 and accuracy metrics were evaluated on the test set, taking for this a minimum probability value, and discarding all those predictions whose probability did not exceed this threshold. The threshold value was varied from 0 to 90, with increments of 10 in each iteration. The global results of each metric were obtained using the macro and weighted averaging schemes on the total of classes, whose graphs are shown in figure 1 and figure 2, respectively.
The results highlight the better general behaviour of the weighted averaging, mainly due to the long-tailed distribution of the dataset (since it assigns a weight according to the number of samples of each class), as well as that a threshold value of 0.7 maintains a compromise between precision and recall.


# Information of input, output data

`/sipecam/ecoinformatica/minikube_sipecam/ecoinf_tests` in sipecamdata server.

# Docker image 

`sipecam/ecoinf-kale-gpu:0.5.0`
