// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

/// Shared error type to use across file processing operations
enum FileProcessingError: Error, LocalizedError {
    case missingFile(String)
    case fileIOError(Error)
    case invalidPath
    case unsupportedFileExtension(String)
    case decompressionFailed(String)

    var errorDescription: String? {
        switch self {
            case let .missingFile(name):
                return "Missing file: \(name)"
            case let .fileIOError(error):
                return "File I/O error: \(error.localizedDescription)"
            case .invalidPath:
                return "Invalid file path"
            case let .unsupportedFileExtension(ext):
                return "Unsupported file extension: \(ext)"
            case let .decompressionFailed(reason):
                return "Decompression failed: \(reason)"
        }
    }
}
