// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class TextEditorViewController: BaseEditorViewController {
    override func viewDidLoad() {
        super.viewDidLoad()

        // Set specific accessibility label for text files
        textView.accessibilityLabel = "Text Editor"

        // Customize for text editing
        textView.autocorrectionType = .no
        textView.autocapitalizationType = .none
        textView.smartDashesType = .no
        textView.smartQuotesType = .no

        // Add line numbers button to toolbar
        if let toolbarItems = toolbar.items {
            let lineNumbersButton = UIBarButtonItem(
                image: UIImage(systemName: "list.number"),
                style: .plain,
                target: self,
                action: #selector(toggleLineNumbers)
            )
            // Create a new array with the line numbers button
            var newItems = Array(toolbarItems)
            newItems.insert(lineNumbersButton, at: newItems.count - 1)
            toolbar.items = newItems
        }
    }

    override func loadFileContent() {
        // Use the base implementation for loading text files
        super.loadFileContent()
    }

    // MARK: - Additional Text Editor Features

    @objc private func toggleLineNumbers() {
        // This would implement line numbers in a production app
        // For now, just show an alert that this feature is coming
        presentAlert(
            title: "Line Numbers",
            message: "Line numbers feature will be available in a future update."
        )

        // Provide haptic feedback
        HapticFeedbackGenerator.generateHapticFeedback(style: .medium)
    }
}
