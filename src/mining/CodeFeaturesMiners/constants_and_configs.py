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
