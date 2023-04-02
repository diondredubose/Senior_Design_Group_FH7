import os
import zipfile
import socket

from UNETv3_TrainFunc import TrainingLoop

host_ip = "192.168.86.24"

# Function to send a message to the host server
def communication_send(message_to_send):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host_ip, 22))
        print('Connected to', host_ip)
        message = message_to_send
        s.sendall(message.encode())
    s.close
    return

# Function to receive a message from the host server
def communication_rec():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 1024))
        s.listen()
        print('waiting for host message at {}:{}...'.format("0.0.0.0", 1024))
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            data = conn.recv(1024)
            print('Received data:', data.decode())
            received_msg = data.decode()
    s.close
    return received_msg

LOAD_DIR = "."

def main():
    print("running nano.py")

    # Establish a connection with the host server and send an acknowledgment
    print("sending communication ack.")
    communication_send("connected")
    print("connection established and confirmed")

    # Start the federated learning loop
    for i in range(2):
        print("federated learning loop #", i)

        # Wait for the host server to send the global model
        print("waiting for model to be sent")
        while(True):
            if(communication_rec() == "model_sent"):
                break

        # Send the file size of the received global model to the host server for confirmation
        file_size = os.path.getsize(r"{}/global_model.zip".format(LOAD_DIR))
        communication_send("{}".format(file_size))

        # Wait for the host server to send the "start_train" message
        print("waiting for \"start_train\" function")
        while(True):
            if(communication_rec() == "start_train"):
                break

        # Unzip the received global model
        with zipfile.ZipFile("{}/{}".format(LOAD_DIR, "global_model.zip"), 'r') as zip_ref:
            zip_ref.extractall("{}/".format(LOAD_DIR))
            os.remove("{}/{}".format(LOAD_DIR, "global_model.zip"))

        # Train the local model using the 'TrainingLoop' function
        print("starting to train")
        model = TrainingLoop(2, 1, .0001)

        # Zip the trained local model and send a message to the host server indicating that training is finished
        with zipfile.ZipFile("trained_model.zip", mode='w') as archive:
            archive.write(r"UNET_MBIMAGENET.pth")
        print("training finished")
        communication_send("train_finish")

        # Wait for the host server to send a message indicating that the trained local model was transferred back
        print("waiting for model to be transferred back")
        while(True):
            if(communication_rec() == "model_sent_back"):
                break

        # Send the file size of the trained local model to the host server for confirmation
        file_size = os.path.getsize(r"{}/trained_model.zip".format(LOAD_DIR))
        communication_send("{}".format(file_size))

        # Remove the local model files from the directory
        os.remove("{}/{}".format(LOAD_DIR, "UNET_MBIMAGENET.pth"))
        os.remove("{}/{}".format(LOAD_DIR, "trained_model.zip"))

    return

# Execute the main function
if __name__ == "__main__":
    main()
