#!/bin/bash
echo "Syncing files to $1"
rsync -av --exclude='.git*' --exclude='sync_to.sh' --filter='dir-merge,- .gitignore' ../Canvas "$1"
