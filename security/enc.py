from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os

def encrypt(key, plaintext):
    iv = os.urandom(16)
    # Ensure the key and IV are the correct length
    if len(key) not in {16, 24, 32}:
        raise ValueError("Key must be 16, 24, or 32 bytes long")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes long")

    # Pad the plaintext to be a multiple of the block size
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(plaintext) + padder.finalize()

    # Create a cipher object using the key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Encrypt the padded plaintext
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext

def decrypt(key, ciphertext):
    # Split the IV and ciphertext
    iv = ciphertext[:16]
    ciphertext = ciphertext[16:]

    # Ensure the key and IV are the correct length
    if len(key) not in {16, 24, 32}:
        raise ValueError("Key must be 16, 24, or 32 bytes long")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes long")

    # Create a cipher object using the key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the plaintext
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return plaintext

# Example usage
key = os.urandom(32)  # 256-bit key
plaintext = b"Hello, World!"

ciphertext = encrypt(key, plaintext)
print("Ciphertext:", ciphertext)

decrypted_plaintext = decrypt(key, ciphertext)
print("Decrypted plaintext:", decrypted_plaintext)
