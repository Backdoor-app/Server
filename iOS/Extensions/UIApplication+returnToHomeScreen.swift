// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

extension UIApplication {
    /// Returns from the foreground app to the home screen.
    func returnToHomeScreen() {
        LSApplicationWorkspace.default()
            .openApplication(withBundleID: "com.apple.springboard")
    }
}
