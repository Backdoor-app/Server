// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import ZIPFoundation

enum FileOperationError: Error {
    case fileNotFound(String)
    case invalidDestination(String)
    case unknownError(String)
    case permissionDenied(String)
}

class FileOperations {
    static let fileManager = FileManager.default

    static func copyFile(at sourceURL: URL, to destinationURL: URL) throws {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw FileOperationError.fileNotFound("Source file not found at \(sourceURL.path)")
        }
        do {
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
        } catch {
            throw FileOperationError.unknownError("Failed to copy file: \(error.localizedDescription)")
        }
    }

    static func moveFile(at sourceURL: URL, to destinationURL: URL) throws {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw FileOperationError.fileNotFound("Source file not found at \(sourceURL.path)")
        }
        do {
            try fileManager.moveItem(at: sourceURL, to: destinationURL)
        } catch {
            throw FileOperationError.unknownError("Failed to move file: \(error.localizedDescription)")
        }
    }

    static func deleteFile(at fileURL: URL) throws {
        guard fileManager.fileExists(atPath: fileURL.path) else {
            throw FileOperationError.fileNotFound("File not found at \(fileURL.path)")
        }
        do {
            try fileManager.removeItem(at: fileURL)
        } catch {
            throw FileOperationError.unknownError("Failed to delete file: \(error.localizedDescription)")
        }
    }

    static func renameFile(at sourceURL: URL, to destinationURL: URL) throws {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw FileOperationError.fileNotFound("File not found at \(sourceURL.path)")
        }
        do {
            try fileManager.moveItem(at: sourceURL, to: destinationURL)
        } catch {
            throw FileOperationError.unknownError("Failed to rename file: \(error.localizedDescription)")
        }
    }

    static func createDirectory(at directoryURL: URL) throws {
        do {
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        } catch {
            throw FileOperationError.unknownError("Failed to create directory: \(error.localizedDescription)")
        }
    }

    static func fileExists(at path: String) -> Bool {
        return fileManager.fileExists(atPath: path)
    }

    static func fileSize(at fileURL: URL) -> UInt64? {
        do {
            let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
            return attributes[.size] as? UInt64
        } catch {
            return nil
        }
    }

    static func creationDate(at fileURL: URL) -> Date? {
        do {
            let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
            return attributes[.creationDate] as? Date
        } catch {
            return nil
        }
    }

    static func unzipFile(at sourceURL: URL, to destinationURL: URL) throws {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw FileOperationError.fileNotFound("File not found at \(sourceURL.path)")
        }
        do {
            try fileManager.unzipItem(at: sourceURL, to: destinationURL)
        } catch {
            throw FileOperationError.unknownError("Failed to unzip file: \(error.localizedDescription)")
        }
    }

    static func hexEditFile(at fileURL: URL, in viewController: UIViewController) {
        guard fileManager.fileExists(atPath: fileURL.path) else {
            print("File not found at \(fileURL.path)")
            return
        }
        let hexEditor = HexEditorViewController(fileURL: fileURL)
        viewController.navigationController?.pushViewController(hexEditor, animated: true)
    }

    static func openIPA(at fileURL: URL, in viewController: UIViewController) {
        guard fileManager.fileExists(atPath: fileURL.path) else {
            print("IPA not found at \(fileURL.path)")
            return
        }
        let ipaEditor = IPAEditorViewController(fileURL: fileURL)
        viewController.navigationController?.pushViewController(ipaEditor, animated: true)
    }
}
