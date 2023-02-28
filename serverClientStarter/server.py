import paramiko
import socket

# create SSH server
server = paramiko.Transport(('localhost', 22))
server.set_gss_host(socket.getfqdn(""))
server.load_server_moduli()
server.add_server_key(paramiko.RSAKey.generate(2048))

# wait for connections
server.start_server()

# accept a client connection
client = server.accept()

# receive the file from the client
filename = client.recv(1024).decode()
with open(filename, "wb") as f:
    while True:
        data = client.recv(1024)
        if not data:
            break
        f.write(data)

# close the client and server connections
client.close()
server.close()
