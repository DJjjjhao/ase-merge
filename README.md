

# **Merge Conflict Resolution: Classification or Generation?**

MergeGen is a generation-based merge conflict resolution approach, which first proposes a structural representation for the merge conflicts and then proposes an encoder-decoder generative model to produce the resolutions through generation. In this repository, we provide our replication package.



## **Environment**

<!-- We offer a `packages.txt` file which lists all the necessary packages for replication. To establish a conda environment and install the required packages, you can execute the subsequent commands. -->

We provide a file named `packages.txt` that contains a list of all the essential packages required for replication. To create a conda environment and install the necessary packages, you can execute the following commands.

```sh
$ conda create -n ase-merge python=3.8

$ conda activate ase-merge

$ python -m pip install -r packages.txt
```



## **Dataset**

Before executing the model, we need to pre-process the dataset first. Since pre-processing is time-consuming,  we use  the `subprocess` module of `python` to prepare the dataset parallelly. The maximum number of processes executing simultaneously is 100, and each subprocess will deal with 1,000 data of the dataset in our experiment. The raw data we use to pre-process is established by the previous work [1] and can be found in their provided [zenodo repository](https://zenodo.org/record/6366908). You can download the raw data and put it in the `RAW_DATA` folder. Then, you can execute the following command to obtain the processed dataset.

```sh
$ python runtotal_dataset.py dataset_parallel.py
```

The file `runtotal_dataset.py` starts the parent process to prepare the dataset parallelly and assigns the sub-tasks to different subprocesses. The file `dataset_parallel.py` is the program executed in the subprocesses and used to prepare the dataset. The processed split dataset will be stored in the folder `PROCESSED`.



## **Model**

<!-- Our model is based on [CodeT5](https://huggingface.co/Salesforce/codet5-small)  and we define our model in file `run_mergegen.py`. If you want to train the model, you can run -->
Our model is built upon [CodeT5](https://huggingface.co/Salesforce/codet5-small), and it is defined in the `run_mergegen.py` file. To fine-tune the model, execute the following command:

```sh
$ python run_mergegen.py train
```

<!-- the model will be saved as `best_model.pt`, and the train process will be saved in the folder `OUTPUT`. -->

The resulting model will be saved as `best_model.pt`, and the training process will be stored in the `OUTPUT` folder.

To test the model on the testing set, use the following command

```sh
$ python run_mergegen.py test
```

<!-- and the resolutions and training/test process will be saved in the folder `OUTPUT`.  -->
The resolutions and testing process will be stored in the `OUTPUT` folder. 

<!-- When train or test the model first time, the file `dataset.py` will be executed, and the split dataset will be merged to obtain the final prepared dataset. -->
When you train or test the model for the first time, the `dataset.py` file will be executed, and the split dataset will be combined to obtain the final prepared dataset.

## **Output**

The `OUTPUT` directory holds the conflict resolutions that were produced by MergeGen.


## **Reference**
[1] Svyatkovskiy A, Fakhoury S, Ghorbani N, et al. Program merge conflict resolution via neural transformers[C]//Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 2022: 822-833.