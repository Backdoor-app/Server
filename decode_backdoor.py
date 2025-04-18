from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.x509 import load_der_x509_certificate, NameOID
from pathlib import Path
import hashlib
import math
import os
import re

# Hardcoded secret key
SECRET = b'bdg_was_here_2025_backdoor_245'
KEY = hashlib.sha256(SECRET).digest()

# Custom permutation
def permute(block):
    return block[::-1]

# Transformation function
def F(data, round_key):
    return hashlib.sha256(data + round_key).digest()[:8]

# Decrypt block
def decrypt_block(block, key):
    R, L = block[:8], block[8:]
    for round in range(3, -1, -1):
        round_key = hashlib.sha256(key + round.to_bytes(4, 'big')).digest()[:8]
        F_val = F(L, round_key)
        new_L = bytes(a ^ b for a, b in zip(R, F_val))
        R, L = L, new_L
    return L + R

# Decrypt data
def decrypt_data(encrypted_data, key, original_len):
    blocks = [encrypted_data[i:i+16] for i in range(0, len(encrypted_data), 16)]
    decrypted_blocks = [decrypt_block(permute(block), key) for block in blocks]
    decrypted_data = b''.join(decrypted_blocks)
    return decrypted_data[:original_len]

# Sanitize filename by removing or replacing invalid characters
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace invalid characters with underscore

# Get certificate name from backdoor file
def get_cert_name_from_backdoor(backdoor_path):
    with open(backdoor_path, "rb") as f:
        cert_len = int.from_bytes(f.read(4), "big")
        cert_der = f.read(cert_len)
        certificate = load_der_x509_certificate(cert_der)
        cert_name = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value if certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME) else Path(backdoor_path).parent.name  # Use parent folder name
        return sanitize_filename(cert_name)

# Decode backdoor file
def decode_backdoor(backdoor_path, output_base_dir):
    try:
        with open(backdoor_path, "rb") as f:
            # Read certificate
            cert_len = int.from_bytes(f.read(4), "big")
            cert_der = f.read(cert_len)
            certificate = load_der_x509_certificate(cert_der)

            # Get certificate name
            cert_name = get_cert_name_from_backdoor(backdoor_path)

            # Read and decrypt .p12 data
            p12_len = int.from_bytes(f.read(4), "big")
            encrypted_p12_len = math.ceil(p12_len / 16) * 16
            encrypted_p12 = f.read(encrypted_p12_len)
            p12_data = decrypt_data(encrypted_p12, KEY, p12_len)

            # Read and decrypt input data
            data_len = int.from_bytes(f.read(4), "big")
            encrypted_data_len = math.ceil(data_len / 16) * 16
            encrypted_data = f.read(encrypted_data_len)
            data = decrypt_data(encrypted_data, KEY, data_len)

            # Read signature
            sig_len = int.from_bytes(f.read(4), "big")
            signature = f.read(sig_len)

        # Verify signature
        public_key = certificate.public_key()
        public_key.verify(
            signature,
            data,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        print(f"Signature verified successfully for certificate {cert_name}!")

        # Create subfolder for this certificate in extracted_output
        cert_output_dir = output_base_dir / cert_name
        cert_output_dir.mkdir(exist_ok=True, parents=True)

        # Save decrypted files
        output_p12_path = cert_output_dir / f"{cert_name}_extracted_certificate.p12"
        output_data_path = cert_output_dir / f"{cert_name}_extracted.mobileprovision"

        with open(output_p12_path, "wb") as f:
            f.write(p12_data)
        with open(output_data_path, "wb") as f:
            f.write(data)
        
        print(f"Extracted .p12 to {output_p12_path}")
        print(f"Extracted input data to {output_data_path}")

        return data, p12_data, cert_name

    except Exception as e:
        print(f"Verification failed for {backdoor_path}: {e}")
        return None, None, None

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    output_dir = base_dir / "extracted_output"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Look for backdoor files in backdoor_output subfolders
    backdoor_dir = base_dir / "backdoor_output"
    if backdoor_dir.exists():
        backdoor_files = list(backdoor_dir.rglob("*_signed_bundle.backdoor"))  # Recursive glob for all subfolders
        for backdoor_file in backdoor_files:
            decode_backdoor(backdoor_file, output_dir)
    else:
        print("No backdoor_output directory found.")