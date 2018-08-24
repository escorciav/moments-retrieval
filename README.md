# Corpus Video Moment Retrieval

Welcome to the Corpus Video Moment Retrieval porject!

## Setup

Let's say you wanna run our codebase. You will need the following requirements:

- Linux box, x64.

- conda.

    In case, it's your first time with conda. You can do the following:

    ```
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    ```

    __Note__: This will install miniconda3 in your $HOME folder. Edit the last part as u prefer 😉.

Once, you are ready just type:

`conda env create -n pytorch -f environment-devel.yml`

In case you wanna do experiments involving `spacy` e.g. intersecting YFCC100M and DiDemo, You may also need the following packages <sup>1</sup>.

`python -m spacy download en_core_web_sm`

That's all. You have all requirements to run the code.

> <sup>1</sup> Sorry, I could not find a way to pack spacy models inside conda.

### Testing

We have a full test-suite in case you want to double check that everything is placed correctly.

First, let's copy the same repo of data to ensure the algorithm look at the same trend.

`bash scripts/setup_data.sh`

I hope it wasn't difficult to reach this point. Now, you are ready to run a full test suite.

`bash scripts/test_all.sh`

__FAQs__

- unable to find conda

    Ensure that you install conda properly. Once, you are confident, check out the comment related to the setup of conda in `scripts/test_all.sh`.

- Memory error

    The models and batch-size are low by default. Maybe, you are trying to use the GPU of someone else 🙂. Edit the `gpu_device` in the file `scripts/test_all.sh`.

Finally, if you are struggling and have spend a couple of hours without seeing the light, drop a line to @escorcia (victor.escorcia@kaust.edu.sa) and @brussel (brussel@adobe.com).