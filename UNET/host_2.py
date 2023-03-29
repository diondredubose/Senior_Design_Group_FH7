import socket
import zipfile
import paramiko
import scp
import os
# import torch

nano_ip = "192.168.86.33"
host_ip = "192.168.86.24"

username = "nano"
password = "12345678"
global_model_path = r"D:\synology\SynologyDrive\Classwork\Diondre\Senior_Design_Group_FH7\UNET\global_model.zip"
remote_model_path = "~/srdsg"


def send_global_model():
    print("entering global model")
    # create a new SSH client object
    ssh = paramiko.SSHClient()

    # automatically add the host key
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("before connection")
    # connect to the Jetson Nano via SSH
    ssh.connect(hostname=nano_ip, username=username, password=password)
    print("after connection")

    # create a new SCP object
    client = scp.SCPClient(ssh.get_transport())
    # use the SCP object to transfer the file
    client.put(global_model_path, remote_path=remote_model_path)
    # close the SCP and SSH connections
    client.close()
    ssh.close()

    return

def send_message(ip_address, port, message):
    # create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # connect to the specified IP address and port
    sock.connect((ip_address, port))
    
    # send the message
    sock.sendall(message.encode())
    
    # close the socket
    sock.close()


def receive_message(ip_address, port):
    # create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # bind the socket to the specified IP address and port
    sock.bind((ip_address, port))
    
    # listen for incoming connections
    sock.listen(1)
    
    # accept the first incoming connection
    conn, addr = sock.accept()
    
    # receive the message
    data = conn.recv(1024)
    
    # close the connection and socket
    conn.close()
    sock.close()
    
    # return the message as a string
    return data.decode()

def main():
    send_message(nano_ip, 8000, "connected")
    print("sent message")
    
    while True:
        if receive_message(host_ip, 9000) == "ACK":
            break

    print("ACK successful")


    with zipfile.ZipFile("global_model.zip", mode = 'w') as archive:
        archive.write(r"D:\synology\SynologyDrive\Classwork\Diondre\Senior_Design_Group_FH7\UNET\UNET_MBIMAGENET.pth")
    
   
    # send global models
    print("Sending global models to the nanos...")
    send_global_model()
    print("here")

    y = 7
    return





if __name__ == "__main__":
    main()