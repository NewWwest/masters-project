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

from src.utils.utils import info_log

class LocalStorageProxy:
    def __init__(self, placeholder_for_bucket_name=None) -> None:
        pass


    def check_exists(self, name, directory ='zipped-results'):
        return os.path.exists(os.path.join(directory, name))



    def upload_files_as_zip(self, file_list, zip_name, directory ='zipped-results'):
        with ZipFile(os.path.join(directory, zip_name), 'w') as zip_f:
            for fn in file_list:
                zip_f.write(fn)

        for fn in file_list:
            os.remove(fn)

        info_log('Zipped files for', zip_name)


    def create_file(self, destination_blob_name, data, directory ='results'):
        destination_blob_name = os.path.join(directory, destination_blob_name)
        with open(destination_blob_name, 'w') as f:
            json.dump(data, f)

        info_log('Saved file', destination_blob_name )
        

