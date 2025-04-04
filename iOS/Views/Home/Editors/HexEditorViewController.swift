// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// Editor for binary files in hexadecimal format
class HexEditorViewController: BaseEditorViewController {
    /// Maximum size in bytes for display in the editor (1MB)
    private let maxDisplaySize: UInt64 = 1_000_000

    override func viewDidLoad() {
        super.viewDidLoad()

        // Set specific accessibility label for hex editor
        textView.accessibilityLabel = "Hex Editor"

        // Configure text view for hex editing
        textView.autocorrectionType = .no
        textView.autocapitalizationType = .none
        textView.smartDashesType = .no
        textView.smartQuotesType = .no
        textView.keyboardType = .asciiCapable

        // Add hex-specific toolbar buttons
        if let toolbarItems = toolbar.items {
            // Toggle between hex and ASCII view
            let viewModeButton = UIBarButtonItem(
                image: UIImage(systemName: "textformat"),
                style: .plain,
                target: self,
                action: #selector(toggleViewMode)
            )

            // Byte count info button
            let infoButton = UIBarButtonItem(
                image: UIImage(systemName: "info.circle"),
                style: .plain,
                target: self,
                action: #selector(showFileInfo)
            )

            // Create a new array with the hex-specific buttons
            var newItems = Array(toolbarItems)
            newItems.insert(viewModeButton, at: newItems.count - 1)
            newItems.insert(infoButton, at: newItems.count - 1)
            toolbar.items = newItems
        }
    }

    override func loadFileContent() {
        // Check file size before loading to avoid performance issues with large files
        do {
            let fileAttributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            guard let fileSize = fileAttributes[FileAttributeKey.size] as? UInt64 else {
                throw NSError(domain: "FileAttributeError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not determine file size"])
            }

            if fileSize > maxDisplaySize {
                presentAlert(
                    title: "Large File Warning",
                    message: "This file is \(ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file)). Only the first \(ByteCountFormatter.string(fromByteCount: Int64(maxDisplaySize), countStyle: .file)) will be displayed for performance reasons."
                )
            }

            // Load the file using our specialized hex loading method
            loadHexContent(maxSize: maxDisplaySize)

        } catch {
            presentAlert(title: "Error", message: "Could not access file: \(error.localizedDescription)")
            Debug.shared.log(message: "File access error: \(error.localizedDescription)", type: .error)
        }
    }

    override func saveChanges() {
        guard let text = textView.text else { return }

        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.center = view.center
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                // Parse hex content
                let hexValues = text.components(separatedBy: CharacterSet.whitespaces).compactMap { UInt8($0, radix: 16) }
                let data = Data(hexValues)

                // Write to file
                try data.write(to: self.fileURL, options: .atomic)

                DispatchQueue.main.async {
                    self.hasUnsavedChanges = false
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()
                    HapticFeedbackGenerator.generateNotificationFeedback(type: .success)
                    self.presentAlert(title: "Success", message: "File saved successfully.")
                }
            } catch {
                DispatchQueue.main.async {
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()
                    self.presentAlert(
                        title: "Error",
                        message: "Could not save file: \(error.localizedDescription)"
                    )
                    Debug.shared.log(message: "Hex save error: \(error.localizedDescription)", type: .error)
                }
            }
        }
    }

    // MARK: - Hex-specific functionality

    /// Flag to control whether we're in hex or ASCII view mode
    private var inHexMode = true

    /// Loads binary content as hex string
    /// - Parameter maxSize: Maximum number of bytes to load
    private func loadHexContent(maxSize: UInt64) {
        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.center = view.center
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                let fileHandle = try FileHandle(forReadingFrom: self.fileURL)
                defer { fileHandle.closeFile() }

                // Only read up to maxSize bytes
                let data = fileHandle.readData(ofLength: Int(maxSize))
                let hexString = data.map { String(format: "%02x", $0) }.joined(separator: " ")

                DispatchQueue.main.async {
                    self.textView.text = hexString
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()

                    // Show byte count info
                    if let fileAttributes = try? FileManager.default.attributesOfItem(atPath: self.fileURL.path),
                       let fileSize = fileAttributes[FileAttributeKey.size] as? UInt64
                    {
                        // If we truncated the file, show a footer with info
                        if fileSize > maxSize {
                            let infoLabel = UILabel()
                            infoLabel.text = "Showing \(data.count) bytes of \(fileSize) total"
                            infoLabel.textAlignment = .center
                            infoLabel.textColor = .secondaryLabel
                            infoLabel.font = .systemFont(ofSize: 12)
                            infoLabel.translatesAutoresizingMaskIntoConstraints = false
                            self.view.addSubview(infoLabel)

                            NSLayoutConstraint.activate([
                                infoLabel.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 10),
                                infoLabel.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -10),
                                infoLabel.bottomAnchor.constraint(equalTo: self.toolbar.topAnchor, constant: -5),
                            ])
                        }
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()
                    self.presentAlert(
                        title: "Error",
                        message: "Failed to load file: \(error.localizedDescription)"
                    )
                    Debug.shared.log(message: "Hex load error: \(error.localizedDescription)", type: .error)
                }
            }
        }
    }

    /// Toggle between hex and ASCII view modes
    @objc private func toggleViewMode() {
        guard let hexText = textView.text else { return }

        inHexMode = !inHexMode

        if inHexMode {
            // Convert ASCII to hex
            let asciiBytes = hexText.data(using: String.Encoding.ascii) ?? Data()
            let hexString = asciiBytes.map { String(format: "%02x", $0) }.joined(separator: " ")
            textView.text = hexString
        } else {
            // Convert hex to ASCII
            let hexValues = hexText.components(separatedBy: CharacterSet.whitespaces).compactMap { UInt8($0, radix: 16) }
            let data = Data(hexValues)
            let asciiText = String(data: data, encoding: String.Encoding.ascii) ?? ""
            textView.text = asciiText
        }

        // Provide feedback about mode change
        self.presentAlert(
            title: inHexMode ? "Hex Mode" : "ASCII Mode",
            message: "Switched to \(inHexMode ? "hexadecimal" : "ASCII") view."
        )

        HapticFeedbackGenerator.generateHapticFeedback(style: .medium)
    }

    /// Display file information
    @objc private func showFileInfo() {
        do {
            let fileAttributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)

            var infoText = ""

            if let fileSize = fileAttributes[FileAttributeKey.size] as? UInt64 {
                infoText += "Size: \(ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file))\n"
            }

            if let creationDate = fileAttributes[FileAttributeKey.creationDate] as? Date {
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                dateFormatter.timeStyle = .medium
                infoText += "Created: \(dateFormatter.string(from: creationDate))\n"
            }

            if let modificationDate = fileAttributes[FileAttributeKey.modificationDate] as? Date {
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                dateFormatter.timeStyle = .medium
                infoText += "Modified: \(dateFormatter.string(from: modificationDate))\n"
            }

            if let fileType = fileAttributes[FileAttributeKey.type] as? String {
                infoText += "Type: \(fileType)\n"
            }

            // Get some info about the hex content
            if let text = textView.text {
                let hexValues = text.components(separatedBy: CharacterSet.whitespaces).compactMap { UInt8($0, radix: 16) }
                infoText += "Bytes in editor: \(hexValues.count)"
            }

            self.presentAlert(title: "File Information", message: infoText)

        } catch {
            self.presentAlert(title: "Error", message: "Could not retrieve file information: \(error.localizedDescription)")
        }
    }

    // Override find and replace to handle hex values correctly
    override func findAndReplace(findText: String, replaceText: String) {
        guard !findText.isEmpty, let text = textView.text else { return }

        if inHexMode {
            // In hex mode, ensure both strings are valid hex
            let findHexValues = findText.components(separatedBy: CharacterSet.whitespaces).compactMap { UInt8($0, radix: 16) }
            let replaceHexValues = replaceText.components(separatedBy: CharacterSet.whitespaces).compactMap { UInt8($0, radix: 16) }

            if findHexValues.isEmpty || replaceHexValues.isEmpty {
                self.presentAlert(title: "Invalid Hex", message: "Please enter valid hexadecimal values.")
                return
            }

            // Convert find/replace strings to consistent format
            let formattedFindText = findHexValues.map { String(format: "%02x", $0) }.joined(separator: " ")
            let formattedReplaceText = replaceHexValues.map { String(format: "%02x", $0) }.joined(separator: " ")

            textView.text = text.replacingOccurrences(of: formattedFindText, with: formattedReplaceText, options: .caseInsensitive)
        } else {
            // In ASCII mode, perform normal text replacement
            textView.text = text.replacingOccurrences(of: findText, with: replaceText, options: .caseInsensitive)
        }

        hasUnsavedChanges = true
    }
}
