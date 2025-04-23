import socket
# import subprocess

# server = subprocess.Popen(
#     ["go", "run", "main.go"],
#     stdin=subprocess.DEVNULL,
#     stdout=subprocess.DEVNULL,
#     stderr=subprocess.DEVNULL,
# )

# def terminate_server():
#     server.terminate()
#     server.wait()

def receive_until_delimiter(client_socket, delimiter='\x00'):
    buffer = ''
    while True:
        part = client_socket.recv(1).decode('utf-8')
        buffer += part
        if delimiter in part:
            break
    return buffer

# Define server address and port
server_address = "localhost"
server_port = 8080

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect((server_address, server_port))
print(f"Connected to server at {server_address}:{server_port}")

def InitUser(username, password) -> None:
    message = '\x01' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent InitUser message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("User initialized: "):
        print("InitUser response received successfully.")
    elif response.startswith("User initialization failed: "):
        raise Exception(response)
    else:
        raise Exception("Unexpected response from server.")
    
def GetUser(username, password) -> None:
    message = '\x02' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent InitUser message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("User retrieved: "):
        print("GetUser response received successfully.")
    elif response.startswith("User retrieving failed: "):
        raise Exception(response)
    else:
        raise Exception("Unexpected response from server.")
    
def StoreFile(username, password, filename, filedata) -> None:
    message = '\x03' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + filedata + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent StoreFile message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("File stored: "):
        print("StoreFile response received successfully.")
    elif response.startswith("File storing failed: "):
        raise Exception(response)
    else:
        raise Exception("Unexpected response from server.")
    
def AppendToFile(username, password, filename, filedata) -> None:
    message = '\x04' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + filedata + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent AppendToFile message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("File appended: "):
        print("AppendToFile response received successfully.")
    elif response.startswith("File appending failed: "):
        raise Exception(response)
    else:
        raise Exception("Unexpected response from server.")

def LoadFile(username, password, filename) -> str:
    message = '\x05' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent LoadFile message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("File loading failed: "):
        raise Exception(response)                
    else:
        print("LoadFile response received successfully.")
        print("File content:", response)
        return response
    
def CreateInvitation(username, password, filename, receipient) -> str:
    message = '\x06' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + receipient.ljust(32, '\x01') + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent CreateInvitation message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("Invitation creation failed: "):
        raise Exception(response)
    else:
        print("CreateInvitation response received successfully.")
        print("Invitation ID:", response)
        return response
    
def AcceptInvitation(username, password, sender, filename, invitation_id) -> None:
    message = '\x07' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + sender.ljust(32, '\x01') + filename.ljust(32, '\x01') + invitation_id.ljust(32, '\x01') + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent AcceptInvitation message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("Invitation accepted: "):
        print("AcceptInvitation response received successfully.")
    elif response.startswith("Invitation acceptance failed: "):
        raise Exception(response)
    else:
        raise Exception("Unexpected response from server.")
    
def RevokeAccess(username, password, filename, receipient) -> None:
    message = '\x08' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + receipient.ljust(32, '\x01') + '\x00'
    client_socket.sendall(message.encode('utf-8'))
    print(f"Sent RevokeAccess message: {message}")
    response = receive_until_delimiter(client_socket)
    print("Received response:", response)
    if response.startswith("Access revoked: "):
        print("RevokeAccess response received successfully.")
    elif response.startswith("Access revoking failed: "):
        raise Exception(response)
    else:
        raise Exception("Unexpected response from server.")
    
# Test the functions
if __name__ == "__main__":
    try:
        # Test InitUser
        username = "test_user"
        password = "test_password"
        message = '\x01' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent InitUser message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("User initialized: "):
            print("InitUser response received successfully.")
        elif response.startswith("User initialization failed: "):
            raise Exception(response)
        else:
            raise Exception("Unexpected response from server.")
        
        # Test GetUser
        username = "test_user"
        password = "test_password"
        message = '\x02' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent InitUser message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("User retrieved: "):
            print("GetUser response received successfully.")
        elif response.startswith("User retrieving failed: "):
            raise Exception(response)
        else:
            raise Exception("Unexpected response from server.")
        
        # Test StoreFile
        username = "test_user"
        password = "test_password"
        filename = "test_file.txt"
        filedata = "Hello, this is a test file."
        message = '\x03' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + filedata + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent StoreFile message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("File stored: "):
            print("StoreFile response received successfully.")
        elif response.startswith("File storing failed: "):
            raise Exception(response)
        else:
            raise Exception("Unexpected response from server.")
        
        # Test AppendToFile
        for i in range(10):
            username = "test_user"
            password = "test_password"
            filename = "test_file.txt"
            filedata = "Hello, this is a test file."
            message = '\x04' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + filedata + '\x00'
            client_socket.sendall(message.encode('utf-8'))
            print(f"Sent AppendToFile message: {message}")
            response = receive_until_delimiter(client_socket)
            print("Received response:", response)
            if response.startswith("File appended: "):
                print("AppendToFile response received successfully.")
            elif response.startswith("File appending failed: "):
                raise Exception(response)
            else:
                raise Exception("Unexpected response from server.")
        
        # Test LoadFile
        username = "test_user"
        password = "test_password"
        filename = "test_file.txt"
        message = '\x05' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent LoadFile message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("File loading failed: "):
            raise Exception(response)                
        else:
            print("LoadFile response received successfully.")
            print("File content:", response)

        username = "test_user "
        password = "test_password "
        InitUser(username, password)
        username2 = "test_user2 "
        password2 = "test_password2 "
        InitUser(username2, password2)
        filename = "test_file.txt "
        filedata = "Hello, this is a test file."
        StoreFile(username, password, filename, filedata)

        # Test CreateInvitation
        receipient = username2
        filename = "test_file.txt "
        message = '\x06' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + receipient.ljust(32, '\x01') + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent CreateInvitation message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("Invitation creation failed: "):
            raise Exception(response)
        else:
            print("CreateInvitation response received successfully.")
            print("Invitation ID:", response)

        # Test AcceptInvitation
        invitation_id = response
        message = '\x07' + username2.ljust(32, '\x01') + password2.ljust(32, '\x01') + username.ljust(32, '\x01') + filename.ljust(32, '\x01') + invitation_id.ljust(32, '\x01') + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent AcceptInvitation message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("Invitation accepted: "):
            print("AcceptInvitation response received successfully.")
        elif response.startswith("Invitation acceptance failed: "):
            raise Exception(response)
        else:
            raise Exception("Unexpected response from server.")
        
        # Test RevokeAccess
        receipient = username2
        filename = "test_file.txt "
        message = '\x08' + username.ljust(32, '\x01') + password.ljust(32, '\x01') + filename.ljust(32, '\x01') + receipient.ljust(32, '\x01') + '\x00'
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent RevokeAccess message: {message}")
        response = receive_until_delimiter(client_socket)
        print("Received response:", response)
        if response.startswith("Access revoked: "):
            print("RevokeAccess response received successfully.")
        elif response.startswith("Access revoking failed: "):
            raise Exception(response)
        else:
            raise Exception("Unexpected response from server.")

    except ConnectionRefusedError:
        print("Failed to connect to the server. Is it running?")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the socket
        client_socket.close()
        print("Connection closed.")