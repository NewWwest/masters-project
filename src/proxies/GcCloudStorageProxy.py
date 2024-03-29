# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------
from zipfile import ZipFile
import json
import os
from google.cloud import storage

from src.utils.utils import info_log

class GcCloudStorageProxy:
    def __init__(self, bucket_name) -> None:
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name
        pass


    def check_exists(self, name, directory ='zipped-results'):
        name = directory + '/' + name
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(name)
        return blob.exists()


    def upload_files_as_zip(self, file_list, zip_name, directory ='zipped-results'):
        with ZipFile(zip_name, 'w') as zip_f:
            for fn in file_list:
                zip_f.write(fn)

        bucket = self.storage_client.bucket(self.bucket_name)

        destination_zip_name = directory + '/' + zip_name
        blob = bucket.blob(destination_zip_name)
        blob.upload_from_filename(zip_name)
        os.remove(zip_name)
        for fn in file_list:
            os.remove(fn)

        info_log('Zipped files for', zip_name)


    def create_file(self, destination_blob_name, data, directory ='results'):
        destination_blob_name = directory + '/' + destination_blob_name

        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(json.dumps(data))

        info_log('Saved file', destination_blob_name )
        

