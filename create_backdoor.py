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

# Load .p12 file data
def load_p12(p12_path):
    try:
        with open(p12_path, "rb") as f:
            p12_data = f.read()
        private_key, certificate, _ = load_key_and_certificates(p12_data, None)  # No password needed
        cert_name = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value if certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME) else Path(p12_path).stem  # Use filename if no common name
        sanitized_cert_name = sanitize_filename(cert_name)
        return private_key, certificate, p12_data, sanitized_cert_name
    except Exception as e:
        print(f"Error loading .p12 file {p12_path}: {e}")
        return None, None, None, None

# Create backdoor file for specific certificate and mobileprovision in BDG cert
def create_backdoor(certs_dir, output_base_dir):
    # Define specific certificate and mobileprovision names
    cert_names = ["HDFC V1", "HDFC V2", "HDFC V3"]
    
    for cert_name in cert_names:
        p12_file = certs_dir / f"{cert_name}.p12"
        mobileprovision_file = certs_dir / f"{cert_name}.mobileprovision"

        if not p12_file.exists() or not mobileprovision_file.exists():
            print(f"Warning: Missing files for {cert_name} - p12: {p12_file}, mobileprovision: {mobileprovision_file}")
            continue

        try:
            private_key, certificate, p12_data, sanitized_cert_name = load_p12(p12_file)
            if not all([private_key, certificate, p12_data, sanitized_cert_name]):
                continue

            with open(mobileprovision_file, "rb") as f:
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

            # Create subfolder for this certificate
            cert_output_dir = output_base_dir / sanitized_cert_name
            cert_output_dir.mkdir(exist_ok=True, parents=True)

            # Create output filename
            output_filename = f"{sanitized_cert_name}_signed_bundle.backdoor"
            output_path = cert_output_dir / output_filename

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

            print(f"Created encrypted .backdoor file at {output_path} with certificate {sanitized_cert_name}")

        except Exception as e:
            print(f"Error creating .backdoor for {p12_file} and {mobileprovision_file}: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    certs_dir = base_dir / "BDG cert"
    output_dir = base_dir / "backdoor_output"
    output_dir.mkdir(exist_ok=True, parents=True)

    create_backdoor(certs_dir, output_dir)