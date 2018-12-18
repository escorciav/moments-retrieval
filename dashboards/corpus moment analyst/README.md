# Dashboard Corpus Moment Analyst

Web based application for visualization of retrieval results.

## Setup
An Anaconda environment is provided by mean of a *.yml* file in the README of the repository. 
Follow those instructions to obtain the virtual environment.


## Data

For this App to be functional the following data is needed.


### 1 - Datasets videos

Donwload the datasets original videos and place them in the respective directory:

`./static/DATA/[DATASET]/ `

where `[DATASET]` is the specific dataset you are interested in.

* **DiDeMo** download [link](), to be placed in folder:     `./static/DATA/didemo/ `
* **Charades** download [link](), to be placed in folder:   `./static/DATA/charades/ `
* **Activtynet** download [link](), to be placed in folder: `./static/DATA/activitynet/ `

> TODO: Add link to the dataset download page.


### 2 - Metadata

The *METADATA* files contains the results outputted by our model. They can be donwloaded [here](https://drive.google.com/open?id=1PNTRukXw-EBFgLekFEJjKxE4PFjHt7Zb).
> TODO: Describe structure of METADATA file.

Download and unzip the files. 
Move the two folders *METADATA* and *duration_metadata* to the folder `./static/`.


## Workflow

To launch a server execute the following command:

```
python server.py --dataset-index [DATASET-INDEX] --number-videos [NUMBER-VIDEOS]
```

where `[DATASET-INDEX]` is the integer value used to select the dataset:

* 0 - Charades
* 1 - DiDeMo
* 2 - Activitynet

and `[NUMBER-VIDEOS]` is the number of videos displayed by the web page.

The server will make available the web pages for the three datasets in different port numbers (which depend on the selected index), such that three instances of the same server can independently coexist on the same machine. 

To reach those web pages open a browser (Chrome is recommended) and type on the address bar the IP address of the machine followed by `:[PORT-NUMBER]` (i.e. `192.168.0.1:60000`). See Note section for more information.

 The `[PORT-NUMBER]` values are:

* 60000 - Charades
* 60001 - DiDeMo
* 60002 - Activitynet

This action will prompt you to the index page in which a table lists all the queries for validation/test sets of the datasets.


![index][index]


Clicking on the query itself you will reach to the result page which will display the ordered results retrieved by the system.


![results][results]

[index]: https://github.com/escorciav/moments-retrieval/blob/collaboration/dashboards/corpus%20moment%20analyst/images/index.png "Index screenshot"
[results]: https://github.com/escorciav/moments-retrieval/blob/collaboration/dashboards/corpus%20moment%20analyst/images/results.png "Results screenshot"


## Note

To obain the IP address for your machine type the following command in the bash:

```bash
hostname -I | awk '{print $1}'
```


