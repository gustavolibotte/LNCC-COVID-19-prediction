#!/bin/bash

for FOLDER in $(ssh joao.miranda@login.sdumont.lncc.br ls /scratch/clickcovid/joao.miranda/logs | tail -171)
do
     rsync -av --exclude "*/post.txt" --exclude "*/post_weights.txt" joao.miranda@login.sdumont.lncc.br:/scratch/clickcovid/joao.miranda/logs/"$FOLDER" sdumont
done
