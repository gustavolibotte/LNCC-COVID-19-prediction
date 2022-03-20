#!/bin/bash

for FOLDER in $(ssh joao.miranda@login.sdumont.lncc.br ls /scratch/clickcovid/joao.miranda/logs | grep "Brazil" | grep fixed | grep 30_day | tail -5)
do
     rsync -av --exclude "*/post.txt" --exclude "*/post_weights.txt" joao.miranda@login.sdumont.lncc.br:/scratch/clickcovid/joao.miranda/logs/"$FOLDER" sdumont
done
