// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

extension DirectoryViewController {
    /// Initialize with a directory URL and title
    /// - Parameters:
    ///   - directoryURL: The URL of the directory to display
    ///   - title: The title to display in the navigation bar
    convenience init(directoryURL: URL, title: String) {
        self.init(directory: directoryURL)
        self.title = title
    }
}
