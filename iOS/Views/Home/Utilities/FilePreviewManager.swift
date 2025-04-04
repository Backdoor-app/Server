// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import QuickLook
import UIKit

class FilePreviewManager {
    /// Present a preview for a file using QuickLook
    /// - Parameters:
    ///   - file: The file to preview
    ///   - viewController: The view controller to present from
    static func presentPreview(for file: File, from viewController: UIViewController) {
        guard FileManager.default.fileExists(atPath: file.url.path) else {
            let alert = UIAlertController(
                title: "File Not Found",
                message: "The file \"\(file.name)\" could not be found.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            viewController.present(alert, animated: true)
            return
        }

        let previewController = FilePreviewController(fileURL: file.url)
        viewController.navigationController?.pushViewController(previewController, animated: true)
    }

    /// Get a list of supported file extensions for preview
    /// - Returns: Array of supported file extensions
    static func supportedFileExtensions() -> [String] {
        return [
            // Documents
            "pdf", "doc", "docx", "pages",
            "xls", "xlsx", "numbers",
            "ppt", "pptx", "key",

            // Images
            "jpg", "jpeg", "png", "gif", "heic", "webp", "tiff", "bmp", "svg",

            // Audio
            "mp3", "m4a", "wav", "aac",

            // Video
            "mp4", "mov", "m4v", "3gp", "avi", "flv", "mpg", "wmv", "mkv",

            // Text and code
            "txt", "rtf", "md", "json", "xml", "csv",
            "swift", "h", "m", "c", "cpp", "py", "js", "html", "css",
        ]
    }

    /// Check if a file can be previewed
    /// - Parameter file: The file to check
    /// - Returns: True if the file can be previewed
    static func canPreview(_ file: File) -> Bool {
        let fileExtension = file.url.pathExtension.lowercased()
        return supportedFileExtensions().contains(fileExtension) ||
            QLPreviewController.canPreview(file.url as QLPreviewItem)
    }
}
