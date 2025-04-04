// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class BaseEditorViewController: UIViewController, UITextViewDelegate {
    /// The URL of the file being edited
    let fileURL: URL

    /// The main text view for editing content
    let textView: UITextView

    /// The toolbar with editing actions
    let toolbar: UIToolbar

    /// Flag indicating if there are unsaved changes
    var hasUnsavedChanges = false

    /// Timer for auto-saving content
    var autoSaveTimer: Timer?

    /// Maximum file size in bytes for direct loading (10MB)
    private let maxDirectLoadSize: UInt64 = 10_000_000

    /// Initializes the editor with a file URL
    /// - Parameter fileURL: The URL of the file to edit
    init(fileURL: URL) {
        self.fileURL = fileURL
        self.textView = UITextView()
        self.toolbar = UIToolbar()
        super.init(nibName: nil, bundle: nil)
        title = fileURL.lastPathComponent
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadFileContent()
        startAutoSaveTimer()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopAutoSaveTimer()
        if hasUnsavedChanges {
            promptSaveChanges()
        }
    }

    // MARK: - Setup

    /// Sets up the user interface
    func setupUI() {
        view.backgroundColor = UIColor.systemBackground.withAlphaComponent(0.95)
        view.layer.applyFuturisticShadow()

        setupTextView()
        setupToolbar()
        setupConstraints()
        setupAccessibility()
    }

    /// Sets up the text view
    private func setupTextView() {
        textView.translatesAutoresizingMaskIntoConstraints = false
        textView.font = .monospacedSystemFont(ofSize: 14, weight: .regular)
        textView.delegate = self
        textView.layer.cornerRadius = 10
        textView.layer.borderColor = UIColor.systemCyan.withAlphaComponent(0.2).cgColor
        textView.layer.borderWidth = 1
        view.addSubview(textView)
    }

    /// Sets up the toolbar and its items
    private func setupToolbar() {
        toolbar.translatesAutoresizingMaskIntoConstraints = false

        let saveButton = UIBarButtonItem(barButtonSystemItem: .save, target: self, action: #selector(saveChanges))
        let copyButton = UIBarButtonItem(barButtonSystemItem: .action, target: self, action: #selector(copyContent))
        let findReplaceButton = UIBarButtonItem(title: "Find/Replace", style: .plain, target: self, action: #selector(promptFindReplace))
        let undoButton = UIBarButtonItem(barButtonSystemItem: .undo, target: self, action: #selector(undoAction))
        let redoButton = UIBarButtonItem(barButtonSystemItem: .redo, target: self, action: #selector(redoAction))
        let flexibleSpace = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)

        toolbar.items = [saveButton, copyButton, findReplaceButton, undoButton, redoButton, flexibleSpace]
        toolbar.tintColor = .systemCyan
        toolbar.layer.cornerRadius = 10
        view.addSubview(toolbar)
    }

    /// Sets up layout constraints
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            textView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            textView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            textView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            textView.bottomAnchor.constraint(equalTo: toolbar.topAnchor, constant: -10),

            toolbar.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            toolbar.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            toolbar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10),
            toolbar.heightAnchor.constraint(equalToConstant: 44),
        ])
    }

    /// Sets up accessibility labels
    private func setupAccessibility() {
        textView.isAccessibilityElement = true
        textView.accessibilityLabel = "File Editor"

        toolbar.isAccessibilityElement = true
        toolbar.accessibilityLabel = "Editor Toolbar"
    }

    // MARK: - File Operations

    /// Loads file content into the text view
    /// Subclasses should override this method to provide specific loading behavior
    func loadFileContent() {
        // Check file size before loading
        do {
            let fileAttributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            guard let fileSize = fileAttributes[.size] as? UInt64 else {
                throw NSError(domain: "FileAttributeError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not determine file size"])
            }

            if fileSize > maxDirectLoadSize {
                presentAlert(
                    title: "Large File Warning",
                    message: "This file is \(ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file)) which may cause performance issues."
                )
            }

            // Default implementation for text files
            loadTextFile()
        } catch {
            presentAlert(title: "Error", message: "Could not load file: \(error.localizedDescription)")
            Debug.shared.log(message: "File load error: \(error.localizedDescription)", type: .error)
        }
    }

    /// Loads a text file into the text view
    private func loadTextFile() {
        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.center = view.center
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                let content = try String(contentsOf: self.fileURL, encoding: .utf8)

                DispatchQueue.main.async {
                    self.textView.text = content
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()
                }
            } catch {
                DispatchQueue.main.async {
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()
                    self.presentAlert(title: "Error", message: "Could not load file: \(error.localizedDescription)")
                    Debug.shared.log(message: "Text file load error: \(error.localizedDescription)", type: .error)
                }
            }
        }
    }

    /// Saves changes to the file
    /// Subclasses should override this method to provide specific saving behavior
    @objc func saveChanges() {
        guard let text = textView.text else { return }

        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.center = view.center
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                try text.write(to: self.fileURL, atomically: true, encoding: .utf8)

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
                    self.presentAlert(title: "Error", message: "Could not save file: \(error.localizedDescription)")
                    Debug.shared.log(message: "File save error: \(error.localizedDescription)", type: .error)
                }
            }
        }
    }

    // MARK: - Editor Actions

    /// Copies the content to the clipboard
    @objc func copyContent() {
        UIPasteboard.general.string = textView.text
        HapticFeedbackGenerator.generateNotificationFeedback(type: .success)
        presentAlert(title: "Copied", message: "Content copied to clipboard.")
    }

    /// Prompts for find and replace
    @objc func promptFindReplace() {
        let alert = UIAlertController(title: "Find and Replace", message: nil, preferredStyle: .alert)

        alert.addTextField { textField in
            textField.placeholder = "Find"
        }

        alert.addTextField { textField in
            textField.placeholder = "Replace"
        }

        let replaceAction = UIAlertAction(title: "Replace", style: .default) { [weak self, weak alert] _ in
            guard let self = self,
                  let findText = alert?.textFields?[0].text,
                  let replaceText = alert?.textFields?[1].text else { return }

            self.findAndReplace(findText: findText, replaceText: replaceText)
        }

        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)

        alert.addAction(replaceAction)
        alert.addAction(cancelAction)

        present(alert, animated: true, completion: nil)
    }

    /// Performs find and replace operation
    /// - Parameters:
    ///   - findText: Text to find
    ///   - replaceText: Text to replace with
    func findAndReplace(findText: String, replaceText: String) {
        guard !findText.isEmpty, let text = textView.text else { return }

        let newText = text.replacingOccurrences(of: findText, with: replaceText, options: .caseInsensitive)
        textView.text = newText
        hasUnsavedChanges = true
    }

    /// Performs undo operation
    @objc func undoAction() {
        textView.undoManager?.undo()
    }

    /// Performs redo operation
    @objc func redoAction() {
        textView.undoManager?.redo()
    }

    // MARK: - Timer Management

    /// Starts the auto-save timer
    func startAutoSaveTimer() {
        autoSaveTimer = Timer.scheduledTimer(
            timeInterval: 60.0,
            target: self,
            selector: #selector(autoSave),
            userInfo: nil,
            repeats: true
        )
    }

    /// Stops the auto-save timer
    func stopAutoSaveTimer() {
        autoSaveTimer?.invalidate()
        autoSaveTimer = nil
    }

    /// Auto-saves changes
    @objc func autoSave() {
        if hasUnsavedChanges {
            saveChanges()
        }
    }

    // MARK: - Alerts

    /// Prompts to save changes
    func promptSaveChanges() {
        let alert = UIAlertController(
            title: "Unsaved Changes",
            message: "Save changes before leaving?",
            preferredStyle: .alert
        )

        let saveAction = UIAlertAction(title: "Save", style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.saveChanges()
            self.navigationController?.popViewController(animated: true)
        }

        let discardAction = UIAlertAction(title: "Discard", style: .destructive) { [weak self] _ in
            self?.navigationController?.popViewController(animated: true)
        }

        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)

        alert.addAction(saveAction)
        alert.addAction(discardAction)
        alert.addAction(cancelAction)

        present(alert, animated: true, completion: nil)
    }

    /// Presents an alert with a title and message
    /// - Parameters:
    ///   - title: Alert title
    ///   - message: Alert message
    func presentAlert(title: String, message: String) {
        let alert = UIAlertController(
            title: title,
            message: message,
            preferredStyle: .alert
        )

        let okAction = UIAlertAction(title: "OK", style: .default)
        alert.addAction(okAction)

        present(alert, animated: true, completion: nil)
    }

    // MARK: - UITextViewDelegate

    func textViewDidChange(_: UITextView) {
        hasUnsavedChanges = true
    }
}
