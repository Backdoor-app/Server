// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import Nuke

// Extension to add Objective-C exposed method for memory warning handling
extension ImageCache {
    @objc func removeAllImages() {
        // Clear all cached images
        ImageCache.shared.clearCache()
    }
}
