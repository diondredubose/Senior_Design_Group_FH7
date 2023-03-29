import socket
import zipfile
import os

nano_ip = "192.168.86.33"
host_ip = "192.168.86.24"

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

    while True:
        if receive_message(nano_ip,8000) == "connected":
            break
    
    print("message recieved")

    send_message(host_ip,9000,"ACK")
    
    while True:
        if receive_message(nano_ip, 8000) == "global model sent":
            break

    LOAD_DIR = "D:\synology\SynologyDrive\Classwork\Diondre\Senior_Design_Group_FH7\UNET"
    file_size = os.path.getsize(r"{}/global_model.zip".format(LOAD_DIR))
    send_message("{}".format(file_size))
    

    
    return



if __name__ == "__main__":
    main()