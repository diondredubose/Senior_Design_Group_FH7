import os
import zipfile
import socket

from UNETv3_TrainFunc import TrainingLoop

host_ip = "192.168.86.24"


def communication_send(message_to_send):
    # create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # connect to the server
        s.connect((host_ip, 22))
        print('Connected to', host_ip)

        # send data to the server
        message = message_to_send
        s.sendall(message.encode())
    s.close
    return

def communication_rec():
    # create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # bind the socket to a specific network interface and port number
        s.bind(("0.0.0.0", 1024))
        # listen for incoming connections
        s.listen()
        print('waiting for host message at {}:{}...'.format("0.0.0.0", 1024))

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

LOAD_DIR = "."
def main():
    print("running nano.py")

    # check and send communication confirmation
    print("sending communication ack.")
    communication_send("connected")
    print("connection established and confirmed")

    for i in range(2):
        print("federated learning loop #", i)
        # wait for files to be sent 
        print("waiting for model to be sent")
        while(True):
            if(communication_rec() == "model_sent"):
                break
        

        file_size = os.path.getsize(r"{}/UNET_MBIMAGENET.pth".format(LOAD_DIR))
        communication_send("{}".format(file_size))
        # send file size to host for file size confirmation
        # unzip
       


        print("waiting for \"start_train\" function")
        while(True):
            if(communication_rec() == "start_train"):
                break
        
        # unzip file

        print("starting to train")
        # start training, call train python file
        model = TrainingLoop(2, 1, .0001)
        print("training finished")
        communication_send("train_finish")
    
    return


if __name__ == "__main__":
    main()
