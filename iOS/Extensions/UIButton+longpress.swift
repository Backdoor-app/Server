// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

extension UIButton {
    private enum AssociatedKeys {
        static var longPressGestureRecognizer = "longPressGestureRecognizer"
    }

    var longPressGestureRecognizer: UILongPressGestureRecognizer? {
        get {
            withUnsafePointer(to: AssociatedKeys.longPressGestureRecognizer) {
                objc_getAssociatedObject(self, $0) as? UILongPressGestureRecognizer
            }
        }
        set {
            withUnsafePointer(to: AssociatedKeys.longPressGestureRecognizer) {
                objc_setAssociatedObject(self, $0, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            }
        }
    }
}
