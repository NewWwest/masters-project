files_to_ignore = ['gradle.properties', 'dockerfile', 'gemfile', 'README', '.gitignore']
extensions_to_ignore = ['.md', '.json' , '.txt', '.gradle', '.sha', '.lock', '.ruby-version', '.yaml', '.yml', '.xml', '.html', '.gitignore', '.d.ts']
cwe_titles_temp = [
    'Cross-site Scripting',
    'SQL Injection',
    'Improper Neutralization',
    'Improper Validation',
    'OS Command Injection',
    'Path Traversal',
    'Cross-Site Request Forgery',
    'CSRF',
    'Deserialization of Untrusted Data',
    'Command Injection',
    'Remote Execution',
    'Server-Side Request Forgery',
    'SSRF',
    'Code Injection',
    'Denial of Service',
    ' DOS '
]
cwe_titles = [cwe.lower() for cwe in cwe_titles_temp]
security_keywords = ['secur', 'vulnerab', 'exploit', 'certificat','authent', 'author' ]

npm_code  = ['.js', '.jsx', '.ts', '.tsx', ]
mvn_code  = ['.java', '.jnl', ]
pypi_code = ['.py', '.ipynb', ]
code_extensions = {
    'npm':npm_code,
    'mvn':mvn_code,
    'pypi':pypi_code,
}