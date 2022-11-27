# Recalls

This directory contains code to calculate the recalls of the different methods and the scores for the final experiment. Use the `calculate_recall` to generate the data, then proceed with the data in the `analyze_spot_dates` notebook to analyze it with the different classes.

Note, for some of the results one needs to use the VulCurator model. To obtain it, get the docker image provided by the authors: nguyentruongggiang/vfdetector:v1
(https://github.com/ntgiang71096/VFDetector, https://zenodo.org/record/7034132). Then run the container and copy the .sav files from the image. Alternativelly, you can run the model within the image and extract ready results.

To copy the .sav files you can use commands:
docker cp CONTAINER_ID:/VFDetector/model/message_classifier.sav message_classifier.sav
docker cp CONTAINER_ID:/VFDetector/model/patch_ensemble.sav patch_ensemble.sav
docker cp CONTAINER_ID:/VFDetector/model/sap_commit_classifier.sav sap_commit_classifier.sav
docker cp CONTAINER_ID:/VFDetector/model/sap_patch_vulfixminer.sav sap_patch_vulfixminer.sav
docker cp CONTAINER_ID:/VFDetector/model/sap_patch_vulfixminer_finetuned_model.sav sap_patch_vulfixminer_finetuned_model.sav
docker cp CONTAINER_ID:/VFDetector/model/tf_commit_classifier.sav tf_commit_classifier.sav
docker cp CONTAINER_ID:/VFDetector/model/tf_issue_classifier.sav tf_issue_classifier.sav
docker cp CONTAINER_ID:/VFDetector/model/tf_message_classifier.sav tf_message_classifier.sav
docker cp CONTAINER_ID:/VFDetector/model/tf_patch_ensemble.sav tf_patch_ensemble.sav
docker cp CONTAINER_ID:/VFDetector/model/tf_patch_vulfixminer.sav tf_patch_vulfixminer.sav
docker cp CONTAINER_ID:/VFDetector/model/tf_patch_vulfixminer_finetuned_model.sav tf_patch_vulfixminer_finetuned_model.sav