// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

/// Extension for BackdoorAIClient to ensure all async calls are properly awaited
extension BackdoorAIClient {
    // Fixed method to support access to the private member variable
    var modelVersionKey: String {
        return "currentModelVersion"
    }
    
    // Other utility methods can be added here
}
