// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CommonCrypto

/// Helper for basic cryptography operations using native iOS libraries
class CryptoHelper {
    // Singleton instance
    static let shared = CryptoHelper()
    
    private init() {}
    
    // MARK: - Encryption Methods
    
    /// Encrypt data using AES with a password
    /// - Parameters:
    ///   - data: Data to encrypt
    ///   - password: Password for encryption
    /// - Returns: Encrypted data as a base64 string
    func encryptAES(_ data: Data, password: String) -> String? {
        // Generate a key from the password
        guard let key = deriveKeyData(from: password, salt: "backdoorsalt", keyLength: 32) else {
            Debug.shared.log(message: "Key derivation failed for encryption", type: .error)
            return nil
        }
        
        // Generate random IV
        let iv = generateRandomBytes(length: 16)
        
        // Create a mutable data to store the cipher text
        let cipherData = NSMutableData()
        
        // Reserve space for the IV at the beginning
        cipherData.append(iv)
        
        // Create a buffer for the ciphertext
        var bufferSize = data.count + kCCBlockSizeAES128
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        
        // Perform the encryption
        var numBytesEncrypted = 0
        
        let cryptStatus = key.withUnsafeBytes { keyBytes in
            iv.withUnsafeBytes { ivBytes in
                data.withUnsafeBytes { dataBytes in
                    CCCrypt(
                        CCOperation(kCCEncrypt),
                        CCAlgorithm(kCCAlgorithmAES),
                        CCOptions(kCCOptionPKCS7Padding),
                        keyBytes.baseAddress, key.count,
                        ivBytes.baseAddress,
                        dataBytes.baseAddress, data.count,
                        &buffer, bufferSize,
                        &numBytesEncrypted
                    )
                }
            }
        }
        
        // Check encryption status
        if cryptStatus == kCCSuccess {
            // Append the encrypted data to the IV
            cipherData.append(buffer, length: numBytesEncrypted)
            
            // Return as base64 string
            return cipherData.base64EncodedString()
        } else {
            Debug.shared.log(message: "AES encryption failed with error: \(cryptStatus)", type: .error)
            return nil
        }
    }
    
    /// Decrypt data using AES with a password
    /// - Parameters:
    ///   - encryptedBase64: Base64 encoded encrypted data with IV prepended
    ///   - password: Password for decryption
    /// - Returns: Decrypted data
    func decryptAES(_ encryptedBase64: String, password: String) -> Data? {
        // Convert base64 to data
        guard let encryptedData = Data(base64Encoded: encryptedBase64) else {
            Debug.shared.log(message: "Failed to decode base64 data", type: .error)
            return nil
        }
        
        // Ensure we have at least the IV
        guard encryptedData.count > kCCBlockSizeAES128 else {
            Debug.shared.log(message: "Encrypted data too short", type: .error)
            return nil
        }
        
        // Extract IV (first 16 bytes for AES)
        let iv = encryptedData.prefix(kCCBlockSizeAES128)
        let dataToDecrypt = encryptedData.suffix(from: kCCBlockSizeAES128)
        
        // Generate key from password
        guard let key = deriveKeyData(from: password, salt: "backdoorsalt", keyLength: 32) else {
            Debug.shared.log(message: "Key derivation failed for decryption", type: .error)
            return nil
        }
        
        // Create a buffer for the decrypted data
        var bufferSize = dataToDecrypt.count + kCCBlockSizeAES128
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        
        // Perform the decryption
        var numBytesDecrypted = 0
        
        let cryptStatus = key.withUnsafeBytes { keyBytes in
            iv.withUnsafeBytes { ivBytes in
                dataToDecrypt.withUnsafeBytes { dataBytes in
                    CCCrypt(
                        CCOperation(kCCDecrypt),
                        CCAlgorithm(kCCAlgorithmAES),
                        CCOptions(kCCOptionPKCS7Padding),
                        keyBytes.baseAddress, key.count,
                        ivBytes.baseAddress,
                        dataBytes.baseAddress, dataToDecrypt.count,
                        &buffer, bufferSize,
                        &numBytesDecrypted
                    )
                }
            }
        }
        
        // Check decryption status
        if cryptStatus == kCCSuccess {
            return Data(bytes: buffer, count: numBytesDecrypted)
        } else {
            Debug.shared.log(message: "AES decryption failed with error: \(cryptStatus)", type: .error)
            return nil
        }
    }
    
    // MARK: - Helper Methods
    
    /// Generate random bytes
    /// - Parameter length: Number of bytes to generate
    /// - Returns: Data containing random bytes
    private func generateRandomBytes(length: Int) -> Data {
        var randomBytes = [UInt8](repeating: 0, count: length)
        _ = SecRandomCopyBytes(kSecRandomDefault, length, &randomBytes)
        return Data(randomBytes)
    }
    
    /// Derive a key from a password using PBKDF2
    /// - Parameters:
    ///   - password: Source password
    ///   - salt: Salt for key derivation
    ///   - keyLength: Length of key to generate
    ///   - iterations: Number of iterations
    /// - Returns: Derived key data or nil on failure
    private func deriveKeyData(from password: String, salt: String, keyLength: Int, iterations: Int = 4096) -> Data? {
        guard let passwordData = password.data(using: .utf8),
              let saltData = salt.data(using: .utf8) else {
            return nil
        }
        
        // Create a temporary buffer to avoid overlapping access
        var keyBuffer = [UInt8](repeating: 0, count: keyLength)
        
        // Call PBKDF2 function with temporary buffer
        let result = saltData.withUnsafeBytes { saltBytes in
            passwordData.withUnsafeBytes { passwordBytes in
                CCKeyDerivationPBKDF(
                    CCPBKDFAlgorithm(kCCPBKDF2),
                    passwordBytes.baseAddress, passwordData.count,
                    saltBytes.baseAddress, saltData.count,
                    CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256),
                    UInt32(iterations),
                    &keyBuffer, keyLength
                )
            }
        }
        
        // Convert buffer to Data only if successful
        return result == kCCSuccess ? Data(keyBuffer) : nil
    }
    
    // MARK: - Hashing Methods
    
    /// Calculate SHA-256 hash of a string
    /// - Parameter input: String to hash
    /// - Returns: Hex string of the hash
    func sha256(_ input: String) -> String {
        guard let data = input.data(using: .utf8) else {
            return ""
        }
        
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        
        data.withUnsafeBytes { buffer in
            _ = CC_SHA256(buffer.baseAddress, CC_LONG(data.count), &hash)
        }
        
        return hash.map { String(format: "%02x", $0) }.joined()
    }
    
    /// Calculate SHA-512 hash of a string
    /// - Parameter input: String to hash
    /// - Returns: Hex string of the hash
    func sha512(_ input: String) -> String {
        guard let data = input.data(using: .utf8) else {
            return ""
        }
        
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA512_DIGEST_LENGTH))
        
        data.withUnsafeBytes { buffer in
            _ = CC_SHA512(buffer.baseAddress, CC_LONG(data.count), &hash)
        }
        
        return hash.map { String(format: "%02x", $0) }.joined()
    }
    
    /// Calculate HMAC using SHA-256
    /// - Parameters:
    ///   - input: Data to authenticate
    ///   - key: Key for HMAC
    /// - Returns: HMAC result as a hex string
    func hmac(_ input: String, key: String) -> String {
        guard let inputData = input.data(using: .utf8),
              let keyData = key.data(using: .utf8) else {
            return ""
        }
        
        var macOut = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        
        keyData.withUnsafeBytes { keyBytes in
            inputData.withUnsafeBytes { dataBytes in
                CCHmac(
                    CCHmacAlgorithm(kCCHmacAlgSHA256),
                    keyBytes.baseAddress, keyData.count,
                    dataBytes.baseAddress, inputData.count,
                    &macOut
                )
            }
        }
        
        return macOut.map { String(format: "%02x", $0) }.joined()
    }
    
    /// Derive a key from a password
    /// - Parameters:
    ///   - password: Source password
    ///   - salt: Salt for key derivation
    ///   - keyLength: Length of key to generate
    ///   - iterations: Number of iterations
    /// - Returns: Derived key as hex string or nil on failure
    func deriveKey(password: String, salt: String, keyLength: Int = 32, iterations: Int = 10000) -> String? {
        guard let keyData = deriveKeyData(from: password, salt: salt, keyLength: keyLength, iterations: iterations) else {
            return nil
        }
        
        return keyData.map { String(format: "%02x", $0) }.joined()
    }
    
    // MARK: - Certificate Utilities
    
    /// Generate a random symmetric key
    /// - Parameter length: Key length in bytes
    /// - Returns: Random key as hex string
    func generateRandomKey(length: Int = 32) -> String {
        let randomData = generateRandomBytes(length: length)
        return randomData.map { String(format: "%02x", $0) }.joined()
    }
    
    /// Compute the CRC32 checksum of data
    /// - Parameter data: Input data
    /// - Returns: CRC32 checksum
    func crc32(of data: Data) -> UInt32 {
        // CRC-32 lookup table
        let table: [UInt32] = [
            0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
            0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
            0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
            0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
            0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
            0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
            0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
            0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
            0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
            0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
            0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
            0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
            0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
            0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
            0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
            0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
            0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
            0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
            0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
            0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
            0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
            0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
            0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
            0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
            0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
            0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
            0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
            0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
            0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
            0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
            0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
            0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
        ]
        
        var crc: UInt32 = 0xffffffff
        
        data.forEach { byte in
            let index = Int((crc ^ UInt32(byte)) & 0xff)
            crc = (crc >> 8) ^ table[index]
        }
        
        return crc ^ 0xffffffff
    }
}
