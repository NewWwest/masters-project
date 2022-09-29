#!/bin/bash
ZONE=us-central1-a

for (( c=0; c<4; c++ ))
do 
    MY_INSTANCE_NAME="keyword-miner-$c"
    echo "Starting $MY_INSTANCE_NAME"
    sed "s/{number}/$c/g" startup-script.sh > "startup-script-$c.sh"

    gcloud compute instances create $MY_INSTANCE_NAME \
        --image-family=debian-10 \
        --image-project=debian-cloud \
        --machine-type=e2-medium \
        --provisioning-model=SPOT \
        --scopes userinfo-email,cloud-platform \
        --metadata-from-file startup-script="startup-script-$c.sh" \
        --zone $ZONE
done

ZONE=us-west1-b

for (( c=4; c<8; c++ ))
do 
    MY_INSTANCE_NAME="keyword-miner-$c"
    echo "Starting $MY_INSTANCE_NAME"
    sed "s/{number}/$c/g" startup-script.sh > "startup-script-$c.sh"

    gcloud compute instances create $MY_INSTANCE_NAME \
        --image-family=debian-10 \
        --image-project=debian-cloud \
        --machine-type=e2-medium \
        --provisioning-model=SPOT \
        --scopes userinfo-email,cloud-platform \
        --metadata-from-file startup-script="startup-script-$c.sh" \
        --zone $ZONE
done