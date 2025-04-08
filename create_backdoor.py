from OpenSSL import crypto
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_der_x509_certificate
import os
from pathlib import Path

# Function to load the private key, certificate, and raw .p12 data from a .p12 file
def load_p12(p12_path):
    try:
        with open(p12_path, "rb") as f:
            p12_data = f.read()
        # Load PKCS12 with no password (as specified)
        private_key, certificate, _ = load_key_and_certificates(p12_data, None)
        return private_key, certificate, p12_data  # Return raw p12_data too
    except Exception as e:
        print(f"Error loading .p12 file: {str(e)}")
        return None, None, None

# Function to create a .backdoor file
def create_backdoor(p12_path, input_data_path, output_path):
    try:
        # Load the private key, certificate, and raw .p12 data
        private_key, certificate, p12_data = load_p12(p12_path)
        if not private_key or not certificate or not p12_data:
            return

        # Read the input data to sign (e.g., mobileprovision)
        with open(input_data_path, "rb") as f:
            data_to_sign = f.read()

        # Create a signature using the private key (signing only mobileprovision data)
        signature = private_key.sign(
            data_to_sign,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Combine certificate, p12 data, mobileprovision data, and signature into a .backdoor file
        with open(output_path, "wb") as f:
            # Write certificate (DER format)
            cert_der = certificate.public_bytes(serialization.Encoding.DER)
            f.write(len(cert_der).to_bytes(4, "big"))  # Length prefix
            f.write(cert_der)
            
            # Write .p12 data
            f.write(len(p12_data).to_bytes(4, "big"))  # Length prefix
            f.write(p12_data)
            
            # Write original mobileprovision data
            f.write(len(data_to_sign).to_bytes(4, "big"))  # Length prefix
            f.write(data_to_sign)
            
            # Write signature
            f.write(len(signature).to_bytes(4, "big"))  # Length prefix
            f.write(signature)

        print(f"Successfully created {output_path}")

    except Exception as e:
        print(f"Error creating .backdoor: {str(e)}")

# Function to verify and extract from the .backdoor file
def verify_backdoor(backdoor_path, output_mobileprovision_path, output_p12_path):
    try:
        with open(backdoor_path, "rb") as f:
            # Read certificate
            cert_len = int.from_bytes(f.read(4), "big")
            cert_der = f.read(cert_len)
            certificate = load_der_x509_certificate(cert_der)
            
            # Read .p12 data
            p12_len = int.from_bytes(f.read(4), "big")
            p12_data = f.read(p12_len)
            
            # Read original mobileprovision data
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

        # Save the extracted mobileprovision data
        with open(output_mobileprovision_path, "wb") as f:
            f.write(data)
        print(f"Extracted mobileprovision to {output_mobileprovision_path}")

        # Save the extracted .p12 data
        with open(output_p12_path, "wb") as f:
            f.write(p12_data)
        print(f"Extracted .p12 to {output_p12_path}")

        return data, p12_data  # Return both for further use if needed

    except Exception as e:
        print(f"Verification failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    # File paths
    base_dir = Path(__file__).parent
    certs_dir = base_dir / "BDG cert"
    
    p12_file = certs_dir / "certificate.p12"
    mobileprovision_file = certs_dir / "profile.mobileprovision"
    output_file = base_dir / "signed_bundle.backdoor"
    extracted_mobileprovision_file = base_dir / "extracted.mobileprovision"
    extracted_p12_file = base_dir / "extracted_certificate.p12"

    # Create the .backdoor file
    create_backdoor(p12_file, mobileprovision_file, output_file)

    # Verify the .backdoor file and extract both files
    original_data, p12_data = verify_backdoor(output_file, extracted_mobileprovision_file, extracted_p12_file)