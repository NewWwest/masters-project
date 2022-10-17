import uuid
import os
import urllib.parse
import requests
import json

workdir = r'D:\Projects\aaa\results\workdir'

class GumTreeProxy:
    
    @staticmethod
    def get_parsed_code_data(old_file_content, new_file_content, filename):
        uuid_string = str(uuid.uuid4())
        old_file_path = f'{uuid_string}-old-{filename}'

        old_file_full_path = os.path.join(workdir, old_file_path)
        with open(old_file_full_path, 'w') as f:
            f.write(old_file_content)

        new_file_path = f'{uuid_string}-new-{filename}'
        new_file_full_path = os.path.join(workdir, new_file_path)
        with open(new_file_full_path, 'w') as f:
            f.write(new_file_content)

        ast_result_full_path = os.path.join(workdir, f'{uuid_string}-ast.json')
        actions_result_file_path = os.path.join(workdir, f'{uuid_string}-action.json')
        response_status = GumTreeProxy._run_java(old_file_full_path, new_file_full_path, ast_result_full_path, actions_result_file_path)

        if response_status:
            with open(ast_result_full_path, 'r') as f:
                ast = json.load(f)
            with open(actions_result_file_path, 'r') as f:
                act = json.load(f)

            return ast, act
        else:
            return None
        
    
    @staticmethod
    def _run_java(old_file_path, new_file_path, ast_result_full_path, actions_result_file_path):
        params = {
            'file_new':new_file_path,
            'file_old':old_file_path,
            'out_ast':ast_result_full_path,
            'out_action':actions_result_file_path,
        }
        query_string = urllib.parse.urlencode(params)
        url = f'http://localhost:8000/parse?{query_string}'
        response = requests.get(url)
        return response.ok
        
