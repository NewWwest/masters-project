sudo apt-get update
sudo apt-get install -yq git python3 python3-pip python3-distutils
gsutil -m cp \
  "gs://clean-equinox-keyword-test/code/keywords.txt" \
  "gs://clean-equinox-keyword-test/code/main.py" \
  "gs://clean-equinox-keyword-test/code/repos_to_process_{number}.txt" \
  "gs://clean-equinox-keyword-test/code/requirements.txt" \
  .
mv "repos_to_process_{number}.txt" "repos_to_process.txt"
pip3 install -r requirements.txt
echo "Done installing"
python3 main.py &
