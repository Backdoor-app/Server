// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

class BundleIdChecker {
    static func shouldModifyBundleId(originalBundleId: String) async -> Bool {
        do {
            let exists = try await iTunesLookup.checkBundleId(originalBundleId)
            Debug.shared.log(message: "Dynamic Protection: Bundle ID \(originalBundleId) exists on App Store: \(exists)", type: .info)
            return exists
        } catch {
            Debug.shared.log(message: "Dynamic Protection: Failed to check bundle ID \(originalBundleId), applying protection as precaution: \(error.localizedDescription)", type: .error)
            return true
        }
    }
}
