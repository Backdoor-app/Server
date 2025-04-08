from OpenSSL import crypto
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_der_x509_certificate  # Added this import
import os
from pathlib import Path

# Function to load the private key and certificate from a .p12 file
def load_p12(p12_path):
    try:
        with open(p12_path, "rb") as f:
            p12_data = f.read()
        # Load PKCS12 with no password (as specified)
        private_key, certificate, _ = load_key_and_certificates(p12_data, None)
        return private_key, certificate
    except Exception as e:
        print(f"Error loading .p12 file: {str(e)}")
        return None, None

# Function to create a .backdoor file
def create_backdoor(p12_path, input_data_path, output_path):
    try:
        # Load the private key and certificate
        private_key, certificate = load_p12(p12_path)
        if not private_key or not certificate:
            return

        # Read the input data to sign (e.g., mobileprovision)
        with open(input_data_path, "rb") as f:
            data_to_sign = f.read()

        # Create a signature using the private key
        signature = private_key.sign(
            data_to_sign,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Combine certificate, data, and signature into a custom .backdoor format
        with open(output_path, "wb") as f:
            # Write certificate (DER format)
            cert_der = certificate.public_bytes(serialization.Encoding.DER)
            f.write(len(cert_der).to_bytes(4, "big"))  # Length prefix
            f.write(cert_der)
            
            # Write original data
            f.write(len(data_to_sign).to_bytes(4, "big"))  # Length prefix
            f.write(data_to_sign)
            
            # Write signature
            f.write(len(signature).to_bytes(4, "big"))  # Length prefix
            f.write(signature)

        print(f"Successfully created {output_path}")

    except Exception as e:
        print(f"Error creating .backdoor: {str(e)}")

# Function to verify and extract from the .backdoor file
def verify_backdoor(backdoor_path):
    try:
        with open(backdoor_path, "rb") as f:
            # Read certificate
            cert_len = int.from_bytes(f.read(4), "big")
            cert_der = f.read(cert_len)
            certificate = load_der_x509_certificate(cert_der)  # Updated to use the correct function
            
            # Read original data
            data_len = int.from_bytes(f.read(4), "big")
            data = f.read(data_len)
            
            # Read signature
            sig_len = int.from_bytes(f.read(4), "big")
            signature = f.read(sig_len)

        # Verify the signature using the public key from the certificate
        public_key = certificate.public_key()
        public_key.verify(
            signature,
            data,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        print("Signature is valid!")
        return data  # Return the original data if needed

    except Exception as e:
        print(f"Verification failed: {str(e)}")
        return None

if __name__ == "__main__":
    # File paths
    base_dir = Path(__file__).parent
    certs_dir = base_dir / "BDG cert"
    
    p12_file = certs_dir / "certificate.p12"
    mobileprovision_file = certs_dir / "profile.mobileprovision"
    output_file = base_dir / "signed_bundle.backdoor"

    # Create the .backdoor file
    create_backdoor(p12_file, mobileprovision_file, output_file)

    # Verify the .backdoor file
    original_data = verify_backdoor(output_file)
    if original_data:
        with open("extracted.mobileprovision", "wb") as f:
            f.write(original_data)