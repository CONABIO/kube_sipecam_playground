# README

Kubeflow pipelines is on a beta status. This component has version 0.3.0 in dummy cluster of AWS as stated in [upgrade to kpf 0.3.0](https://github.com/e2fyi/kubeflow-aws/commit/18ae2b7c59e255bbe9a2e3ca6354ad85237a058f). The k8s cluster that is not dummy has kubeflow 0.2.4

- Link to [pipelines kubeflow terminology](https://www.kubeflow.org/docs/pipelines/overview/concepts/)

	* A pipeline is a description of a machine learning (ML) workflow, including all of the components in the workflow and how the components relate to each other in the form of a graph.

		* The pipeline configuration includes the definition of the inputs (parameters) required to run the pipeline and the inputs and outputs of each component.
		* When you run a pipeline, the system launches one or more Kubernetes Pods corresponding to the steps (components) in your workflow (pipeline). The Pods start Docker containers, and the containers in turn start your programs.
		* After developing your pipeline, you can upload your pipeline using the Kubeflow Pipelines UI or the Kubeflow Pipelines SDK.

	* A pipeline is a description of an ML workflow, including all of the components that make up the steps in the workflow and how the components interact with each other
	
	* A pipeline component is self-contained set of code that performs one step in the ML workflow (pipeline), such as data preprocessing, data transformation, model training, and so on. A component is analogous to a function, in that it has a name, parameters, return values, and a body

		* A component specification in YAML format describes the component for the Kubeflow Pipelines system.
		* For the complete definition of a component, see the [component specification](https://www.kubeflow.org/docs/pipelines/reference/component-spec/)
		* You must package your component as a Docker image. Components represent a specific program or entry point inside a container.
		* Each component in a pipeline executes independently. The components do not run in the same process and cannot directly share in-memory data. You must serialize (to strings or files) all the data pieces that you pass between the components so that the data can travel over the distributed network. You must then deserialize the data for use in the downstream component.

	* A step is an execution of one of the components in the pipeline. The relationship between a step and its component is one of instantiation, much like the relationship between a run and its pipeline.

- Link to [Introduction to the Pipelines SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/).

	* The Kubeflow Pipelines SDK provides a set of Python packages that you can use to specify and run your machine learning (ML) workflows.
	* Install it via: `pip3 install kfp --upgrade --user`. Then check with `which dsl-compile`.
	* You can use the Kubeflow Pipelines SDK to build machine learning pipelines. You can use the SDK to execute your pipeline, or alternatively you can upload the pipeline to the Kubeflow Pipelines UI for execution
	* Each pipeline is defined as a Python program. Before you can submit a pipeline to the Kubeflow Pipelines service, you must compile the pipeline to an intermediate representation. The intermediate representation takes the form of a YAML file compressed into a .tar.gz file. Use the dsl-compile command to compile the pipeline that you chose

- Link to [kfp package sphix documentation](https://kubeflow-pipelines.readthedocs.io/en/latest/)
