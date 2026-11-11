import sys, paramiko
host, user, pwd = "192.168.10.144", "root", "123456"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, username=user, password=pwd, timeout=10)
sftp = c.open_sftp()
for local, remote in zip(sys.argv[1::2], sys.argv[2::2]):
    sftp.put(local, remote)
    print("put", remote)
sftp.close(); c.close()
