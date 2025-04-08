from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from pathlib import Path
import hashlib
import math

# Hardcoded secret key for encryption (shared with decode script)
SECRET = b'my_custom_secret_key_2023'
KEY = hashlib.sha256(SECRET).digest()

# Padding function to align data to block size
def pad(data, block_size=16):
    if len(data) % block_size == 0:
        return data
    padding_len = block_size - (len(data) % block_size)
    return data + b'\0' * padding_len

# Custom permutation for obfuscation (byte reversal)
def permute(block):
    return block[::-1]

# Transformation function for Feistel network
def F(data, round_key):
    return hashlib.sha256(data + round_key).digest()[:8]

# Custom block encryption using a Feistel network
def encrypt_block(block, key):
    L, R = block[:8], block[8:]
    for round in range(4):
        round_key = hashlib.sha256(key + round.to_bytes(4, 'big')).digest()[:8]
        F_val = F(R, round_key)
        new_R = bytes(a ^ b for a, b in zip(L, F_val))
        L, R = R, new_R
    return R + L

# Encrypt and obfuscate data
def encrypt_data(data, key):
    padded_data = pad(data)
    blocks = [padded_data[i:i+16] for i in range(0, len(padded_data), 16)]
    encrypted_blocks = [permute(encrypt_block(block, key)) for block in blocks]
    return b''.join(encrypted_blocks)

# Load .p12 file data
def load_p12(p12_path):
    try:
        with open(p12_path, "rb") as f:
            p12_data = f.read()
        private_key, certificate, _ = load_key_and_certificates(p12_data, None)
        return private_key, certificate, p12_data
    except Exception as e:
        print(f"Error loading .p12 file: {e}")
        return None, None, None

# Create an encrypted and obfuscated .backdoor file
def create_backdoor(p12_path, input_data_path, output_path):
    try:
        # Load .p12 components
        private_key, certificate, p12_data = load_p12(p12_path)
        if not all([private_key, certificate, p12_data]):
            return

        # Read input data (e.g., .mobileprovision)
        with open(input_data_path, "rb") as f:
            data_to_sign = f.read()

        # Sign the original input data
        signature = private_key.sign(
            data_to_sign,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Encrypt and obfuscate the .p12 and input data
        encrypted_p12 = encrypt_data(p12_data, KEY)
        encrypted_data = encrypt_data(data_to_sign, KEY)

        # Write to .backdoor file
        with open(output_path, "wb") as f:
            cert_der = certificate.public_bytes(serialization.Encoding.DER)
            f.write(len(cert_der).to_bytes(4, "big"))
            f.write(cert_der)
            f.write(len(p12_data).to_bytes(4, "big"))  # Original length
            f.write(encrypted_p12)
            f.write(len(data_to_sign).to_bytes(4, "big"))  # Original length
            f.write(encrypted_data)
            f.write(len(signature).to_bytes(4, "big"))
            f.write(signature)

        print(f"Created encrypted .backdoor file at {output_path}")
    except Exception as e:
        print(f"Error creating .backdoor: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    certs_dir = base_dir / "BDG cert"
    
    p12_file = certs_dir / "certificate.p12"
    input_file = certs_dir / "profile.mobileprovision"
    backdoor_file = base_dir / "signed_bundle.backdoor"

    # Create the .backdoor file
    create_backdoor(p12_file, input_file, backdoor_file)