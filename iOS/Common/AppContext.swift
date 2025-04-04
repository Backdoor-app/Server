// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

/// Represents the current state of the app for AI context
struct AppContext {
    let currentScreen: String
    let additionalData: [String: Any]

    func toString() -> String {
        var result = "Current Screen: \(currentScreen)\n"
        result += "App Data:\n"
        for (key, value) in additionalData {
            result += "- \(key): \(value)\n"
        }
        return result
    }
}
