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
import os
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def info_log(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.info(log_message)


def warn_log(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.warning(log_message)


def get_files_in_from_directory(dir, extension=None, startswith=None):
    files_list = []
    for root, subdirs, files in os.walk(dir):
        for file in files:
            if extension != None and not file.endswith(extension):
                continue

            if startswith != None and not file.startswith(startswith):
                continue
            
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list


def is_hexadecimal_string(val):
    try:
        test = int(val, 16)
        return True
    except:
        return False