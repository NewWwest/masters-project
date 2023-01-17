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
import uuid
import os
import urllib.parse
import requests
import json
import random

parser_url = 'http://localhost:8000'
workdir = 'workdir'

class GumTreeProxy:
    
    @staticmethod
    def get_parsed_code_data(old_file_content, new_file_content, filename):
        uuid_string = str(uuid.uuid4()) + '-' + str(random.randint(0, 2_000_000))

        old_file_path = f'{uuid_string}-old-{filename}'

        old_file_full_path = os.path.join(workdir, old_file_path)
        with open(old_file_full_path, 'w', encoding='UTF-8') as f:
            f.write(old_file_content)

        new_file_path = f'{uuid_string}-new-{filename}'
        new_file_full_path = os.path.join(workdir, new_file_path)
        with open(new_file_full_path, 'w', encoding='UTF-8') as f:
            f.write(new_file_content)

        ast_result_full_path = os.path.join(workdir, f'{uuid_string}-ast.json')
        actions_result_file_path = os.path.join(workdir, f'{uuid_string}-action.json')
        response_status = GumTreeProxy._run_java(old_file_full_path, new_file_full_path, ast_result_full_path, actions_result_file_path)

        if response_status:
            with open(ast_result_full_path, 'r', encoding='UTF-8') as f:
                ast = json.load(f)
            with open(actions_result_file_path, 'r', encoding='UTF-8') as f:
                act = json.load(f)

            GumTreeProxy._try_remove_files(old_file_full_path, new_file_full_path, ast_result_full_path, actions_result_file_path)
            return ast, act
        else:
            GumTreeProxy._try_remove_files(old_file_full_path, new_file_full_path, ast_result_full_path, actions_result_file_path)
            return None
            
    @staticmethod
    def _try_remove_files(old_file_full_path, new_file_full_path, ast_result_full_path, actions_result_file_path):
        try:
            if os.path.exists(old_file_full_path):
                os.remove(old_file_full_path) 
            if os.path.exists(new_file_full_path):
                os.remove(new_file_full_path) 
            if os.path.exists(ast_result_full_path):
                os.remove(ast_result_full_path) 
            if os.path.exists(actions_result_file_path):
                os.remove(actions_result_file_path) 
        except Exception as e:
            print(e)
            print('Cleaning up failed')


    @staticmethod
    def _run_java(old_file_path, new_file_path, ast_result_full_path, actions_result_file_path):
        params = {
            'file_new':new_file_path,
            'file_old':old_file_path,
            'out_ast':ast_result_full_path,
            'out_action':actions_result_file_path,
        }
        query_string = urllib.parse.urlencode(params)
        url = f'{parser_url}/parse?{query_string}'
        response = requests.get(url)
        return response.ok
        
