# Corpus Moment Analyst

Web based application for visualization of retrieval results.

## Introduction
The Corpus Moment Analyst is a web application used for visualizing the results of our video retrieval system. The main page is the *index page* containing a table listing all the queries for the selected dataset. These entries are associated with the rank IoU score computed by the algorithm and can then be sorted as pleased. An explanatory screenshot is shown in the following figure.

![index][index]

By clicking on the query itself you will reach to the *result page* which will display the ordered videos retrieved by the system for that particular query. An example is shown below.

![results][results]

Selecting a video will open the player. This section provides several controls over the video, it is possible to play/pause and activate/deactivate loop over the predicted interval as well as increase and decrease the speed. It is also possible to move the sliders under the video to tune the reproduction interval and inspect the neighbor of the predicted interval. In the end, a reset button can restore the default settings.

![player][player]

[![video](https://github.com/escorciav/moments-retrieval/blob/collaboration/dashboards/corpus%20moment%20analyst/images/results.png)](https://www.youtube.com/watch?v=248S6jQ2wAI)

[index]: https://github.com/escorciav/moments-retrieval/blob/collaboration/dashboards/corpus%20moment%20analyst/images/index.png "Index screenshot"
[results]: https://github.com/escorciav/moments-retrieval/blob/collaboration/dashboards/corpus%20moment%20analyst/images/results.png "Results screenshot"
[player]: https://github.com/escorciav/moments-retrieval/blob/collaboration/dashboards/corpus%20moment%20analyst/images/results.png "Results screenshot"



## Getting started

1. Install all the required dependencies:

	An Anaconda environment is provided by mean of a *.yml* file in the main README of the repository. 
	Follow those instructions to obtain the virtual environment.

2. Download data:

	For this App to be functional the following data is needed.

	- **Datasets videos**

		Download the datasets original videos and place them in the respective directory: `./static/DATA/[DATASET]/ ` where `[DATASET]` is the specific dataset you are interested in.

		* **DiDeMo** download [link](), to be placed in folder:     `./static/DATA/didemo/ `
		* **Charades** download [link](), to be placed in folder:   `./static/DATA/charades/ `
		* **Activtynet** download [link](), to be placed in folder: `./static/DATA/activitynet/ `

		> TODO: Add the link to the datasets download page.

	- **Metadata**

		The *METADATA* files contain the results outputted by our model. They can be downloaded [here](https://drive.google.com/open?id=1PNTRukXw-EBFgLekFEJjKxE4PFjHt7Zb).
		> TODO: Describe the structure of METADATA file.

		Download and unzip the files. 
		Move the two folders *METADATA* and *duration_metadata* to the folder `./static/`.


## Instructions

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

This action will prompt you to the index page. See the introduction to understand how the different pages are linked together.


## Note

To obtain the IP address for your machine type the following command in the bash:

```bash
hostname -I | awk '{print $1}'
```


