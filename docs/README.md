## InterActive Learning Toolkit

Active Learning algorithms such as [ProbCover](https://paperswithcode.com/paper/active-learning-through-a-covering-lens) have achieved state of the art results, and even made improvements over Self-Supervised and Semi-Supervised techniques. However, most of these results are in simualted environments: the datasets were actually labelled, but the labels were hidden from the model until the Active Learning "oracle" unhid them. We wanted to make Active Learning work for you -- dear Reader with a truly unlabelled dataset, by providing a full-service workflow. You start with an image folder of unlabelled data, and use this Toolkit to generate embeddings, select the examples to label, and save the labels.

### When to use this toolkit: 

This toolkit focuses on the Cold Start problem. Many Active Learning frameworks, such as weakly supervised or semi-supervised learning rely on an "initial set" of labelled examples and work to propogate those labels to unlabelled examples. The harder problem is when ALL your data is unlabelled. Where do you even know where to start labelling? That's where this toolkit comes in. We'll help you label as many examples as possible to get a working classifier, and provide guidance on when you can stop labelling

### How to use this toolkit: 

Note: Unfortunately, the interactivity of this notebook tends to slow way down on Colab (passing large amounts of data to the javascript front-end appears to be a known issue). We recommend cloning this repo and running a jupyter notebook locally.

### The steps are as follows:

### Part 1: Prep work
Enter the root directory where your images are stored in the cells below. The cell after that will find all images in the folder, so don't worry about file naming and folder structure

### Part 2: Create Embeddings
Specify or create Embeddings. All our Active Learning algorithms require a good embedding space as a prerequisite. If you already have self-supervised embeddings from your data, simply enter where the npy or pth file is stored. If you don't have embeddings, you have three options -- using the forward pass on a pre-trained VGG model to generate embeddings, using the forward pass on a self-supervised ResNet to generate embeddings, or fine-tuning a self-supervised model on your dataset with SimCLR. The first two will run quickly, the second can take a very long time to run depending on the size of your dataset.

### Part 3: Label your examples with Active Learning
Use your embeddings to select examples to label. Instantiate a class of our Active Learners as specified below, and it will generate a list of examples to label. The step will save a data manifest for your future use

### Part 4: Test a model
Use those labels to build a DataLoader. You're now ready to train a classifier that should have maximum performance per labelled example!

### Citations

Philip Lippe, https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html

Yehuda, Ofer, et al. "Active Learning Through a Covering Lens." arXiv preprint arXiv:2205.11320 (2022).

Akshay L Chandra and Vineeth N Balasubramanian, Deep Active Learning Toolkit for Image Classification in PyTorch. https://github.com/acl21/deep-active-learning-pytorch

Munjal, Prateek, et al. "Towards robust and reproducible active learning using neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

Note: This website is an initial Work In Progress preview of our research and subject to change as the work progresses
