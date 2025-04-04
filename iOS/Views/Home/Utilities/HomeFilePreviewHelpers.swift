// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import QuickLook
import UIKit

// Important helper functions for file previews
extension HomeViewController {
    /// Present a preview for a file
    /// - Parameter file: The file to preview
    public func presentFilePreview(for file: File) {
        // Check if the file exists
        guard FileManager.default.fileExists(atPath: file.url.path) else {
            utilities.handleError(
                in: self,
                error: FileAppError.fileNotFound(file.name),
                withTitle: "File Not Found"
            )
            return
        }

        // Use the preview controller
        let previewController = FilePreviewController(fileURL: file.url)
        navigationController?.pushViewController(previewController, animated: true)
    }

    /// Present an image preview
    /// - Parameter file: The image file to preview
    public func presentImagePreview(for file: File) {
        presentFilePreview(for: file)
    }
}