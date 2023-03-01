import paramiko
import scp
import subprocess
import time
import socket
import os
import torch

nano_ip = "192.168.86.33"
global_model_path = r"D:\synology\SynologyDrive\Classwork\Diondre\WS2\UNET.pth"
remote_model_path = "~/srdsg/Senior_Design_Group_FH7/UNET/"

remote_model_file = "~/srdsg/Senior_Design_Group_FH7/UNET/UNET_MBIMAGENET.pth"
username = "nano"
password = "12345678"
weight_path = r"D:\synology\SynologyDrive\Classwork\Diondre\WS2\weights"

def fed_averager():
    Weights = []
    Global_Model = {}
    for file in os.listdir("{}".format(weight_path)):
        Weights.append(torch.load("{}/{}".format(weight_path, file), map_location=torch.device('cpu')))
        os.remove("{}/{}".format(weight_path, file))
    for i in range(Weights.__len__()):
        for key in Weights[0]:
            temp = Weights[i][key]
            if key in Global_Model:
                Global_Model[key] += temp
            else:
                Global_Model[key] = temp
    for key in Global_Model:
        Global_Model[key] = Global_Model[key] / Weights.__len__()
    torch.save(obj = Global_Model, f= global_model_path)
    

def communication_rec():
    # create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # bind the socket to a specific network interface and port number
        s.bind(("0.0.0.0", 22))
        # listen for incoming connections
        s.listen()
        print('waiting for nano #1 message at {}:{}...'.format("0.0.0.0", 22))

        # accept a nano connection
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)

            # receive data from the nano
            data = conn.recv(1024)
            print('Received data:', data.decode())
            received_msg = data.decode()
    s.close
    return received_msg

def communication_send(message_to_send):
    # create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # connect to the server
        s.connect((nano_ip, 1024))
        print('Connected to', nano_ip)

        # send data to the server
        message = message_to_send
        s.sendall(message.encode())
    s.close
    return


def send_global_model():
    # create a new SSH client object
    ssh = paramiko.SSHClient()

    # automatically add the host key
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # connect to the Jetson Nano via SSH
    ssh.connect(hostname=nano_ip, username=username, password=password)

    # create a new SCP object
    client = scp.SCPClient(ssh.get_transport())
    # use the SCP object to transfer the file
    client.put(global_model_path, remote_path=remote_model_path)
    # close the SCP and SSH connections
    client.close()
    ssh.close()

    return



def retrieve_global_model(): #check to see if works or not
        # create a new SSH client object
    ssh = paramiko.SSHClient()

    # automatically add the host key
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # connect to the Jetson Nano via SSH
    ssh.connect(hostname=nano_ip, username=username, password=password)

    # create a new SCP object
    client = scp.SCPClient(ssh.get_transport())
    # use the SCP object to transfer the file
    client.get(remote_model_file, weight_path)
    # close the SCP and SSH connections
    client.close()
    ssh.close()

    return


def main():
    print("running host.py")
    # connection acknowledgement
    while(True):
        if(communication_rec() == "connected"):
            break
    # loop for multiple federated learning cycles
    for i in range(2):
        print("federated learning loop #", i)
        # send global models
        print("Sending global models to the nanos...")
        send_global_model()
        print("global model sent to all nanos!")

        # send message to nano to start training models
        print("starting to train models...")
        communication_send("start_train")

        # wait for training to be finished
        print("waiting for \"train_done\" message...")
        while(True):
            if(communication_rec() == "train_finish"):
                break
        print("training complete!")

        # pull trained models from nanos
        print("receiving trained models...")
        retrieve_global_model()
        print("trained models downloaded!")

        # aggregate the models together
        print("beginning aggregation of models...")
        fed_averager()
        print("aggregation complete")
    return

if __name__ == "__main__":
    main()













