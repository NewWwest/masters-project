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
files_to_ignore = ['gradle.properties', 'dockerfile', 'gemfile', 'README', '.gitignore']
extensions_to_ignore = ['.md', '.json' , '.txt', '.gradle', '.sha', '.lock', '.ruby-version', '.yaml', '.yml', '.xml', '.html', '.gitignore', '.d.ts']

better_security_keywords = [
    'secur',
    'vulnerab',
    'exploit', 
    'certificat', 
    'authent',
    'leak',
    'sensit',
    'crash',
    'attack',
    'deadlock',
    'segfault',
    'malicious',
    'corrupt',
]

npm_code  = ['.js', '.jsx', '.ts', '.tsx', ]
npm_like_code  = ['.cjs', '.mjs', '.iced', '.liticed', '.coffee', '.litcoffee', '.ls', '.es6', '.es', '.sjs', '.eg']
java_code  = ['.java']
java_like_code  = ['.jnl', '.jar', '.class', '.dpj', '.jsp', '.scala', '.sc', '.kt', '.kts', '.ktm']
pypi_code = ['.py', '.py3']
pypi_code_like = ['.pyw', '.pyx', '.ipynb']

code_extensions = {
    'npm':npm_code,
    'npm_like':npm_like_code,
    'mvn':java_code,
    'mvn_like':java_like_code,
    'pypi':pypi_code,
    'pypi_like':pypi_code_like,
}

separators = ['/','\\','`','*','_','{','}','[',']','(',')','>','#','+','-',',','.','!','$','\'', '\t', '\n', '\r']
