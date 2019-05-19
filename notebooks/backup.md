List models backup in G-Drive as well as strategy used to backup.

# Models

## _Drive_

mcn_41
mcn_42
mcn_43
smcn_47
smcn_48
smcn_49
smcn_50
smcn_51
smcn_52
mcn_54
smcn_53
smcn_54
smcn_55
smcn_56
smcn_40
smcn_42
smcn_43

## _2nd-tier_

[s]mcn_44
[s]mcn_45
smcn_46
mcn_40
smcn_41

# Strategy

1. Move everything to skynet as README is there.

    1. List all files in the folder. For example

        `ls -lhR data/interim/mcn_41/ | less`

    1. Sync files. We only sync new files.

        `rsync --remove-source-files --ignore-existing -RPrlptD data/interim/smcn_40 /home/escorciav/projects/moments-retrieval/workers/skynet-base/`

        What about if we did an in-place change of a file in the machine?

    1. Dump MD5

        > the following command might need a bit of tweaking according to depth of folder.

        `md5sum data/interim/smcn_43/*/*/* data/interim/smcn_43/*/* > ~/projects/moments-retrieval/workers/skynet-base/ibex-smcn_53`

    1. Check MD5

        `md5sum -c`

    1. Backup and document changes.

    1. Delete folder from source.

2. Copy everything to tyler-dev

    1. Sync folders

        `rsync --remove-source-files [list-of-folders] [dest-folders]`

    1. clean there (remove unused things)

3. Backup to Drive

    We use [rclone](https://rclone.org/drive/). To copy a folder, we can use:

    `rclone --dry-run --progress --no-traverse -v copy [dirname] drive-kaust:[drive-endpoint]/[dirname]`

    Use TXT-file with [list of files to transfer](https://rclone.org/filtering/#files-from-read-list-of-source-file-names)

    `rclone --dry-run --progress -v copy --files-from files-from.txt [dirname] drive-kaust:[drive-endpoint]/[dirname]`