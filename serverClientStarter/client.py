import paramiko

# specify the filename of the file to send
filename = "example.zip"

# create SSH client and connect to server
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('localhost', username='tch5', password='Lewandowski#9')

# open an SFTP session on the SSH connection
sftp = client.open_sftp()

# create the destination directory if it doesn't already exist

# send the file to the server
sftp.put("C:/Users/tch5/Desktop/client/example.zip", "C:/Users/tch5/Desktop/server/transferred_hoe.zip")

# close the SFTP session and SSH connection
sftp.close()
client.close()
