// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// Editor for property list (plist) files with syntax highlighting and validation
class PlistEditorViewController: BaseEditorViewController {
    /// Flag to enable plist syntax validation
    private var validateSyntax = true

    override func viewDidLoad() {
        super.viewDidLoad()

        // Set specific accessibility label for plist files
        textView.accessibilityLabel = "Plist Editor"

        // Configure text view for plist editing
        textView.autocorrectionType = .no
        textView.autocapitalizationType = .none
        textView.smartDashesType = .no
        textView.smartQuotesType = .no

        // Add validate button to the toolbar
        if let toolbarItems = toolbar.items {
            let validateButton = UIBarButtonItem(
                image: UIImage(systemName: "checkmark.circle"),
                style: .plain,
                target: self,
                action: #selector(validatePlist)
            )

            // Create a new array with the validate button
            var newItems = Array(toolbarItems)
            newItems.insert(validateButton, at: newItems.count - 1)
            toolbar.items = newItems
        }
    }

    override func loadFileContent() {
        // Use the base implementation but check file type first
        if fileURL.pathExtension.lowercased() != "plist" {
            presentAlert(
                title: "Warning",
                message: "This file doesn't have a .plist extension. It may not be a valid property list file."
            )
        }

        super.loadFileContent()
    }

    override func saveChanges() {
        // Optionally validate before saving
        if validateSyntax {
            if !validatePlistContent() {
                presentAlert(
                    title: "Invalid Plist",
                    message: "The content doesn't appear to be a valid property list. Do you want to save anyway?"
                )
                return
            }
        }

        // Use the base implementation for saving
        super.saveChanges()
    }

    // MARK: - Plist-specific functionality

    /// Validates that the content is a valid property list
    /// - Returns: True if valid, false otherwise
    private func validatePlistContent() -> Bool {
        guard let text = textView.text else { return false }

        // Convert text to data
        guard let data = text.data(using: .utf8) else { return false }

        // Try to parse as property list
        do {
            let _ = try PropertyListSerialization.propertyList(from: data, options: [], format: nil)
            return true
        } catch {
            Debug.shared.log(message: "Plist validation error: \(error.localizedDescription)", type: .error)
            return false
        }
    }

    /// Validates the plist content and shows result
    @objc private func validatePlist() {
        if validatePlistContent() {
            presentAlert(
                title: "Valid Plist",
                message: "The content is a valid property list."
            )
            HapticFeedbackGenerator.generateNotificationFeedback(type: .success)
        } else {
            presentAlert(
                title: "Invalid Plist",
                message: "The content is not a valid property list. Please check for syntax errors."
            )
            HapticFeedbackGenerator.generateNotificationFeedback(type: .error)
        }
    }

    /// Toggle syntax validation on save
    @objc private func toggleValidation() {
        validateSyntax = !validateSyntax
        presentAlert(
            title: "Validation " + (validateSyntax ? "Enabled" : "Disabled"),
            message: validateSyntax ?
                "Plist will be validated before saving." :
                "Plist will be saved without validation."
        )
    }
}
