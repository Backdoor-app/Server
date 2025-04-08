from cryptography import x509  # Updated import
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import os
from pathlib import Path

# Function to verify and decode the .backdoor file
def decode_backdoor(backdoor_path, output_data_path):
    try:
        with open(backdoor_path, "rb") as f:
            # Read certificate
            cert_len = int.from_bytes(f.read(4), "big")
            cert_der = f.read(cert_len)
            certificate = x509.load_der_x509_certificate(cert_der, default_backend())  # Fixed line
            
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
        print("Signature verification successful!")

        # Save the decoded original data
        with open(output_data_path, "wb") as f:
            f.write(data)
        print(f"Successfully decoded data to {output_data_path}")

        return data

    except Exception as e:
        print(f"Decoding/verification failed: {str(e)}")
        return None

if __name__ == "__main__":
    # File paths
    base_dir = Path(__file__).parent
    backdoor_file = base_dir / "signed_bundle.backdoor"
    output_file = base_dir / "decoded.mobileprovision"

    # Decode the .backdoor file
    decode_backdoor(backdoor_file, output_file)