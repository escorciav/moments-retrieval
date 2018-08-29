# Handy condor commands

- List jobs hold

    `condor_q -hold`

- List reasons

    `condor_q -analyze`

- Interactive job

    `condor_submit -interactive`

    - get a particular node

    `condor_submit -interactive requirements='(TARGET.Machine == "ilcomp24.ilcomp")'`

- Status

    `condor_status` or `condor_status -claimed`

    `condor_q -nob -all | sort -k5`

According to @zhding, there are more commands in the cluster wiki.

# Notes

- interactive mode does not setup GPU. Thus, you gotta set the env variable CUDA_VISIBLE_DEVICES by yourself 😫. On the other side, this is a backdoor 😎. BTW, there are a couple of backdoors to exploit.

- conda environments works nicely everywhere in ilcompm[0-1]. That way you don't need to setup/update/push docker images. Once you are done, "dockerization" should be as easy as embeded your conda envinroment inside a docker image.

- does not seems to be a fair-share policy, neither penalties for using the cluster a lot.

- the maximum number of jobs in the queue is updated manually by the sys-admin.

- BTW, you can't put more that the maximum number of jobs into the queue. This didn't make sense to me because that number is set really low. However,this goes in line with the absence of fair-share policy, and the sys-admin adjust the number such that your waiting time is at most 1-2 days.

- What to do if I need more jobs?
I like bash scripting, thus I put a for loop calling my program multiple times passing an index to dump outputs with different names. However, that does not deal with dependencies between jobs. Like, launching more jobs of the same kind after you reach your limit of jobs running. In that case, you gotta write your own scheduler 😫. It could be as simple as `condor_q` and check-out the number of jobs running inside a while loop and sleep. It seems like condor sucks as scheduler, maybe it's not.

- check out @niklaus management tool. I don't know exactly what it does, but looks fancy.
BTW, it's a matter of workflow, we usually believe that a tool could double our productivity, but realistically it may just increase in 2-10%. Summing up over 13 weeks, fancy tools without training can give u an average return of 0-2% in productivity.

# Notes of conduct

It seems that some individuals just sat on top of GPUs to "run experiments later". Deadlines bring out the worse of humans, I guess. If that pisses you off: (i) open tmux, create as many panes as the number of max jobs, set synchronize mode, and launch interactive sessions. You can join them later with condor_ssh_job; or (ii) launch as many notebooks jobs as you can. In any case, take a deep breath and go for a coffee. It's simply out of your control.

_Note_ Fair-share policies are great for these kind of scenarios, punishing irresponsible users silently. Other strategies could be to track GPU-usage with nvidia-smi, and kill jobs that don't use the GPU over a period of 3 hours.