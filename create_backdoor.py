from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from cryptography.x509 import load_der_x509_certificate, NameOID
from pathlib import Path
import hashlib
import math
import os
import re

# Hardcoded secret key
SECRET = b'bdg_was_here_2025_backdoor_245'
KEY = hashlib.sha256(SECRET).digest()

# Padding function
def pad(data, block_size=16):
    if len(data) % block_size == 0:
        return data
    padding_len = block_size - (len(data) % block_size)
    return data + b'\0' * padding_len

# Custom permutation
def permute(block):
    return block[::-1]

# Transformation function
def F(data, round_key):
    return hashlib.sha256(data + round_key).digest()[:8]

# Encrypt block
def encrypt_block(block, key):
    L, R = block[:8], block[8:]
    for round in range(4):
        round_key = hashlib.sha256(key + round.to_bytes(4, 'big')).digest()[:8]
        F_val = F(R, round_key)
        new_R = bytes(a ^ b for a, b in zip(L, F_val))
        L, R = R, new_R
    return R + L

# Encrypt data
def encrypt_data(data, key):
    padded_data = pad(data)
    blocks = [padded_data[i:i+16] for i in range(0, len(padded_data), 16)]
    encrypted_blocks = [permute(encrypt_block(block, key)) for block in blocks]
    return b''.join(encrypted_blocks)

# Sanitize filename by removing or replacing invalid characters
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace invalid characters with underscore

# Get unique filename by appending a counter if the name already exists
def get_unique_filename(base_dir, base_name, extension):
    counter = 1
    original_name = f"{base_name}{extension}"
    unique_name = original_name

    while (base_dir / unique_name).exists():
        unique_name = f"{base_name}_{counter}{extension}"
        counter += 1

    return unique_name

# Load .p12 file data
def load_p12(p12_path):
    try:
        with open(p12_path, "rb") as f:
            p12_data = f.read()
        private_key, certificate, _ = load_key_and_certificates(p12_data, None)  # No password needed
        cert_name = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value if certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME) else "unknown_cert"
        sanitized_cert_name = sanitize_filename(cert_name)  # Sanitize the name
        return private_key, certificate, p12_data, sanitized_cert_name
    except Exception as e:
        print(f"Error loading .p12 file {p12_path}: {e}")
        return None, None, None, None

# Create backdoor file
def create_backdoor(p12_path, input_data_path, output_base_dir):
    try:
        private_key, certificate, p12_data, cert_name = load_p12(p12_path)
        if not all([private_key, certificate, p12_data, cert_name]):
            return

        with open(input_data_path, "rb") as f:
            data_to_sign = f.read()

        # Sign the original input data
        signature = private_key.sign(
            data_to_sign,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Encrypt and obfuscate
        encrypted_p12 = encrypt_data(p12_data, KEY)
        encrypted_data = encrypt_data(data_to_sign, KEY)

        # Create unique output filename with sanitized certificate name
        base_name = cert_name
        extension = "_signed_bundle.backdoor"
        unique_filename = get_unique_filename(output_base_dir, base_name, extension)
        output_path = output_base_dir / unique_filename

        # Write to .backdoor file
        with open(output_path, "wb") as f:
            cert_der = certificate.public_bytes(serialization.Encoding.DER)
            f.write(len(cert_der).to_bytes(4, "big"))
            f.write(cert_der)
            f.write(len(p12_data).to_bytes(4, "big"))
            f.write(encrypted_p12)
            f.write(len(data_to_sign).to_bytes(4, "big"))
            f.write(encrypted_data)
            f.write(len(signature).to_bytes(4, "big"))
            f.write(signature)

        print(f"Created encrypted .backdoor file at {output_path} with certificate {cert_name}")

    except Exception as e:
        print(f"Error creating .backdoor for {p12_path} and {input_data_path}: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    certs_dir = base_dir / "BDG cert"
    output_dir = base_dir / "backdoor_output"
    output_dir.mkdir(exist_ok=True)

    # Handle multiple .p12 and .mobileprovision files
    p12_files = list(certs_dir.glob("*.p12"))
    mobileprovision_files = list(certs_dir.glob("*.mobileprovision"))

    for p12_file in p12_files:
        for mobileprovision in mobileprovision_files:
            create_backdoor(p12_file, mobileprovision, output_dir)