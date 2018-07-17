Handy condor commands

- List jobs hold

    `condor_q -hold`

- List reasons

    `condor_q -analyze`

- Interactive job

    `condor_submit -interactive`

    - get a particular node

    `condor_submit -interactive requirements='(TARGET.Machine == "ilcomp24.ilcomp")'`