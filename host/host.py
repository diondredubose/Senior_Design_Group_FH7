# host.py
# used to control training from host computer to each of the jetson nanos using threading
# for more information, see host_readme.txt

# Created by Benhur Yonas
# Last updated by Benhur Yonas 02/22/2023 12:40am


import threading
import subprocess
import paramiko
import time
import numpy as np

# TODO fix the password issues

# Set up IP addresses of the Jetson Nanos
#jetson_ips = ["192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.4", "192.168.1.5", "192.168.1.6", "192.168.1.7"]
jetson_ips = ["192.168.86.33"]

# Define a function to transfer files to each Jetson Nano using SCP
def scp_transfer(jetson_ip, file_path):
    scp_command = ["scp", file_path, f"nano@{jetson_ip}:~/srdsg/comm_test/"]
    subprocess.run(scp_command)

# Define a function to run federated learning on each Jetson Nano using threading
def run_federated_learning(jetson_ip, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(jetson_ip, username="nano", password=password) 
    stdin, stdout, stderr = ssh.exec_command("python ~/srdsg/comm_test/federated_learning.py") 
    time.sleep(5)  # Wait for command to finish
    ssh.close()

# Define a function to average the model weights
def average_models(jetson_ips, password):
    # Connect to the first Jetson Nano to download the model weights
    download_command = ["scp", f"nano@{jetson_ips[0]}:~/srdsg/comm_test/model_weights.npy", "."]
    subprocess.run(download_command)
    
    # Load the model weights from the first Jetson Nano
    model_weights = np.load("model_weights.npy")

    # Average the model weights from the other Jetson Nanos
    for jetson_ip in jetson_ips[1:]:
        download_command = ["scp", f"nano@{jetson_ip}:~/srdsg/comm_test/model_weights.npy", "."]
        subprocess.run(download_command)
        new_weights = np.load("model_weights.npy")
        model_weights += new_weights

    # Divide the model weights by the number of Jetson Nanos to get the average
    model_weights /= len(jetson_ips)

    # Save the averaged model weights
    np.save("averaged_model_weights.npy", model_weights)
    
    # Transfer the averaged model weights back to each Jetson Nano
    for jetson_ip in jetson_ips:
        scp_command = ["scp", "averaged_model_weights.npy", f"nano@{jetson_ip}:~/srdsg/comm_test/"]
        subprocess.run(scp_command)

# Enter the SSH password for each Jetson Nano
password = "12345678"

# Transfer the initial global models to each Jetson Nano using SCP
data_file_path = "data.csv" # TODO change to make sure all are the same model names
# this will be the pretrained NYU models

print("\n\n\nhost.py\nUsed to train depth estimation models using federated learning\n\n")
total_loops = input("how many federated learning loops would you like to do: ").split()
print("\ntransferring pretrained NYU model to each nano...\n")

for jetson_ip in jetson_ips:
    t = threading.Thread(target=scp_transfer, args=(jetson_ip, data_file_path))
    t.start()

for thread in threading.enumerate():
    if thread is not threading.main_thread():
        thread.join()


print("\npretrained NYU model transferred to all nanos\n")

loop = 1

# TODO make below code into a loop to repeat federated learning
# for loop in total loops
print("\nstarting federated learning loop: ",  loop, "\n")

# Start federated learning on each Jetson Nano using threading
for jetson_ip in jetson_ips:
    t = threading.Thread(target=run_federated_learning, args=(jetson_ip, password))
    t.start()

# Wait for all threads to complete
for thread in threading.enumerate():
    if thread is not threading.main_thread():
        thread.join()


print("\nfederated learning loop #", loop, " finished\n")
print("\naveraging models...\n")
# Average the model weights
average_models(jetson_ips, password)

loop = loop + 1

# end of the loop
