from keystoneauth1 import session
from keystoneauth1.identity import v3
from swiftclient.client import Connection
import os
import sys

# reference: https://docs.openstack.org/python-swiftclient/latest/client-api.html
#
# environment variables must be set for authentication like this in openstack horizon:
# AUTH_SERVER=10.0.87.254,AUTH_PROJECT_NAME=openstack,AUTH_USERNAME=admin,AUTH_PASSWORD=password
#
# environment varialbes set at terminal:
# export AUTH_SERVER=10.0.87.254; export AUTH_PROJECT_NAME=openstack; export AUTH_USERNAME=admin; export AUTH_PASSWORD=password;
#
# example
# python3 upload-files.py /tmp/results/irtt_data.json students-project
#
# or
#
# example
# AUTH_SERVER=10.0.87.254 AUTH_PROJECT_NAME=openstack AUTH_USERNAME=samie AUTH_PASSWORD=password python3 upload-files.py /tmp/results/irtt_data.json container-folder

def main():
    args_num = len(sys.argv)
    if args_num <= 2:
        print("no file path or container name is provided")
        return
    
    full_file_addr = sys.argv[1]
    print(f"file addr to upload: {full_file_addr}")

    container = sys.argv[2]
    print(f"swift container name to upload to: {container}")

    auth_server = os.environ.get('AUTH_SERVER')
    if not auth_server:
        print("no auth server is in environment variables")
        return

    auth_project_name = os.environ.get('AUTH_PROJECT_NAME')
    if not auth_project_name:
        print("no auth project name is in environment variables")
        return 0

    auth_username = os.environ.get('AUTH_USERNAME')
    if not auth_username:
        print("no auth username is in environment variables")
        return 0

    auth_password = os.environ.get('AUTH_PASSWORD')
    if not auth_password:
        print("no auth password is in environment variables")
        return 0

    print(f'Contacting https://{auth_server}:5000/v3/ ...')

    # Create a password auth plugin
    auth = v3.Password(auth_url=f'https://{auth_server}:5000/v3/',
                       username=auth_username,
                       password=auth_password,
                       user_domain_name='Default',
                       project_name=auth_project_name,
                       project_domain_name='Default')

    # Create session
    keystone_session = session.Session(auth=auth)

    # Create swiftclient Connection
    conn = Connection(session=keystone_session)

    file_addr, file_name = os.path.split(full_file_addr)
    with open(full_file_addr, 'rb') as f:
        file_data = f.read()
        conn.put_object(
            container,
            file_name,
            contents=file_data,
        )

main()
