// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

extension UIApplication {
    func topMostViewController() -> UIViewController? {
        guard let windowScene = connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first
        else {
            return nil
        }
        var topController = window.rootViewController
        while let presentedController = topController?.presentedViewController {
            topController = presentedController
        }
        return topController
    }
}

// Define notification names in a central location
// These notification names need to be unique across the app

// This is now defined in NotificationNames to avoid redeclaration issues
// Do not add changeTab here - use NotificationNames.changeTab instead

// Centralized enum for notification names to avoid ambiguity
public enum NotificationNames {
    // Used for tab switching across the app
    static let changeTab = Notification.Name("com.backdoor.notifications.changeTab")
    // Define other notification names here
}
