files_to_ignore = ['gradle.properties', 'dockerfile', 'gemfile', 'README', '.gitignore']
extensions_to_ignore = ['.md', '.json' , '.txt', '.gradle', '.sha', '.lock', '.ruby-version', '.yaml', '.yml', '.xml', '.html', '.gitignore', '.d.ts']
cwe_titles_temp = [
    'Out-of-bounds Write',
    'Use After Free',
    'Out-of-bounds Read',
    'Out of bounds Write',
    'Out of bounds Read',
    'Integer Overflow',

    'Improper Neutralization',
    'Improper Input Neutralization',
    'Improper Validation',
    'Improper Input Validation',
    'Missing Authentication',
    'Improper Authentication',
    'Incomplete Authentication'
    'Missing Authorization',
    'Improper Authorization',
    'Incomplete Authorization'

    'Missing Restriction',
    'Improper Restriction',
    'Incorrect Default Permissions',

    'Hard-coded Credentials',
    'Hardcoded Credentials',
    'Hard coded Credentials',

    'OS Command Injection',
    'OSCI',
    'Cross-site Scripting',
    'XSS',
    'Crosssite Scripting',
    'SQL Injection',

    'Uncontrolled Resource Consumption',
    'Incorrect Default Permissions',
    
    'Path Traversal',
    'Cross-Site Request Forgery',
    'CrossSite Request Forgery',
    'CSRF',
    'Deserialization of Untrusted Data',
    'Command Injection',
    'Remote Execution',
    'Server-Side Request Forgery',
    'ServerSide Request Forgery',
    'SSRF',
    'Code Injection',
    'Denial of Service',
    ' DOS ',
]
cwe_titles = [cwe.lower() for cwe in cwe_titles_temp]
security_keywords = ['secur', 'vulnerab', 'exploit', 'certificat', 'authent', 'author' ]

npm_code  = ['.js', '.jsx', '.ts', '.tsx', ]
mvn_code  = ['.java', '.jnl', ]
pypi_code = ['.py', '.ipynb', ]
code_extensions = {
    'npm':npm_code,
    'mvn':mvn_code,
    'pypi':pypi_code,
}