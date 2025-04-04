// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

extension UIView {
    /// Find the view controller that contains this view
    func findViewController() -> UIViewController? {
        var responder: UIResponder? = self
        while let nextResponder = responder?.next {
            responder = nextResponder
            if let viewController = responder as? UIViewController {
                return viewController
            }
        }
        return nil
    }
}

extension UILabel {
    /// Add padding to a UILabel
    var padding: UIEdgeInsets {
        get {
            return .zero
        }
        set {
            let paddingView = UIView(frame: CGRect(
                x: 0, y: 0,
                width: newValue.left + newValue.right,
                height: newValue.top + newValue.bottom)
            )
            paddingView.backgroundColor = .clear
            
            self.bounds = self.bounds.inset(by: newValue.inverted())
            self.frame = CGRect(
                x: self.frame.origin.x - newValue.left,
                y: self.frame.origin.y - newValue.top,
                width: self.frame.size.width + newValue.left + newValue.right,
                height: self.frame.size.height + newValue.top + newValue.bottom
            )
        }
    }
}

extension UIEdgeInsets {
    func inverted() -> UIEdgeInsets {
        return UIEdgeInsets(
            top: -top,
            left: -left,
            bottom: -bottom,
            right: -right
        )
    }
}

// Extension removed to avoid conflict with UIApplication+TopViewController.swift
// The topMostViewController() implementation is now centralized in UIApplication+TopViewController.swift
