import os

if os.path.exists('./public-ipv4') and os.path.exists('./token'):

    ip = open('./public-ipv4').read()
    token = open('./token').read()
    token = token.split(':')[3]

    print('\n    Based on auto-detection of your EC2 public IP, use the following URL:\n\n        http://%s:%s' % (ip, token))

print('\n    To shut down Jupyter notebook use the command: pkill jupyter')
