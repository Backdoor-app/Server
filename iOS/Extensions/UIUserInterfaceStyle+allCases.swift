// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

extension UIUserInterfaceStyle: @retroactive CaseIterable {
    public static var allCases: [UIUserInterfaceStyle] = [.unspecified, .dark, .light]
    var description: String {
        switch self {
            case .unspecified:
                return "System"
            case .light:
                return "Light"
            case .dark:
                return "Dark"
            @unknown default:
                return "Unknown Mode"
        }
    }
}
