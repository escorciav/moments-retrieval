# dashboards

The window to your data and model results ...

## Setup

For simplicity, we use the same development environment of the main project. That leaves room for shrinking the actual requirements if required on "production".

## Workflow

To launch a server just execute the following commands __from the dashboard folder__

```bash
source setup.sh
source [X]
```

e.g. if you wanna launch the moment retrieval demo replace `[X]` by `moment_retrieval_demo.sh`.

> Make sure that you edit the `setup.sh` file accordingly if you have an exotic config.

By default, the ports are:

2002: moment explorer of DiDeMo val with results of ICCV17 MCN model.

2005: sentence retrieval in DiDeMo val

2006: moment retrieval in DiDeMo val with ICCV17 MCN model.

## Nitty-gritty

### sentence retrieval

It showcases an exhaustive demo of Corpus Video Sentence Retrieval.

This demo relies on:

- JSON with results from corpus video sentence retrieval.

    By August 2018, we were still iterating with REST API 😓. Take a look at this [notebook](https://git.corp.adobe.com/escorcia/moments-retrieval/blob/adobe/notebooks/7-breakdown-per-noun.ipynb) for more details about how we dumped the features. Hopefully, the variable names are indicative enough to guide you without having a cerebral embolism, only headache and anger 😅.

    In particular, search for the section `Sentence retrieval evaluation`.

    > Sorry, I could find a way to point to the section, despite of the fact that I added a reference to it.

    The current demo version display the results of a SMCN model trained on DiDeMo videos, with ResNet features, using only Local features, and without using Intra-loss (yes, there are more hyper-parameters 😁).

### moment retrieval

It showcases an exhaustive demo of Corpus Video Moment Retrieval.

This demo relies on:

- GIFs files of the validation set.

- HDF5 with visual embedded representation for all possible moments in DiDeMo validation corpus (rgb and flow embedded moments).

- HDF5 with weights of MCN ICCV-2017 model (rgb and flow).

### Didemo explorer

Showcases Corpus Video Moment Retrieval in the context of the ground-truth annotations.

This demo relies on:

- GIFs files of the validation set.

- JSON with results from corpus video moment retrieval.

    By August 2018, we were still iterating with REST API 😓. Take a look at this [notebook](https://git.corp.adobe.com/escorcia/moments-retrieval/blob/adobe/notebooks/10-retrieval-over-corpus.ipynb) for more details about how we dumped the features. Hopefully, the variable names are indicative enough to guide you without having a cerebral embolism, only headache and anger 😅.

    In particular, search for the section `Data for dashboard backend`.

    > Sorry, I could find a way to point to the section, despite of the fact that I added a reference to it.

    The current demo version display the results of MCN ICCV-2017 model (rgb and flow).