// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class LicenseViewController: UIViewController {
    var textContent: String?
    var titleText: String?

    override func viewDidLoad() {
        super.viewDidLoad()
        title = titleText
        let textView = UITextView()
        textView.text = textContent
        textView.isEditable = false
        textView.translatesAutoresizingMaskIntoConstraints = false

        let monospacedFont = UIFont.monospacedSystemFont(ofSize: 12.0, weight: .regular)
        textView.font = monospacedFont

        // Scroll to top
        textView.setContentOffset(CGPoint.zero, animated: true)
        textView.contentInset = UIEdgeInsets(top: 16, left: 16, bottom: 16, right: 16)

        view.addSubview(textView)

        NSLayoutConstraint.activate([
            textView.topAnchor.constraint(equalTo: view.topAnchor),
            textView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            textView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            textView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }
}
