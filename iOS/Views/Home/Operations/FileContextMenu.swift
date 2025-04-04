// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import ZIPFoundation

class FileContextMenu: NSObject, UIContextMenuInteractionDelegate {
    // MARK: - Properties

    /// The view controller that owns this menu
    private weak var viewController: UIViewController?

    /// The file this menu is for
    private var file: File

    /// Callback for when a file action is performed
    var onActionPerformed: (() -> Void)?

    // MARK: - Initialization

    /// Initialize with a file and view controller
    /// - Parameters:
    ///   - file: The file to create a menu for
    ///   - viewController: The view controller that owns this menu
    init(for file: File, in viewController: UIViewController) {
        self.file = file
        self.viewController = viewController
        super.init()
    }

    // MARK: - UIContextMenuInteractionDelegate

    func contextMenuInteraction(_: UIContextMenuInteraction, configurationForMenuAtLocation _: CGPoint) -> UIContextMenuConfiguration? {
        return UIContextMenuConfiguration(identifier: nil, previewProvider: nil) { [weak self] _ in
            guard let self = self else { return nil }
            return self.createContextMenu()
        }
    }

    // MARK: - Menu Creation

    /// Creates the context menu for the file
    /// - Returns: A UIMenu with actions appropriate for the file
    private func createContextMenu() -> UIMenu {
        var actions: [UIAction] = []

        // Open action - always available
        actions.append(UIAction(title: "Open", image: UIImage(systemName: "arrow.up.forward.app"), handler: { [weak self] _ in
            guard let self = self, let viewController = self.viewController as? HomeViewController else { return }
            viewController.openFile(self.file)
        }))

        // Share action - always available
        actions.append(UIAction(title: "Share", image: UIImage(systemName: "square.and.arrow.up"), handler: { [weak self] _ in
            guard let self = self, let viewController = self.viewController else { return }
            self.shareFile(viewController)
        }))

        // Rename action - always available
        actions.append(UIAction(title: "Rename", image: UIImage(systemName: "pencil"), handler: { [weak self] _ in
            guard let self = self, let viewController = self.viewController as? HomeViewController else { return }
            viewController.renameFile(self.file)
        }))

        // Type-specific actions
        if file.isDirectory {
            // Directory-specific actions
            actions.append(UIAction(title: "Compress", image: UIImage(systemName: "archivebox"), handler: { [weak self] _ in
                guard let self = self, let viewController = self.viewController else { return }
                self.compressDirectory(viewController)
            }))
        } else {
            // File-specific actions

            // Compress action
            actions.append(UIAction(title: "Compress", image: UIImage(systemName: "archivebox"), handler: { [weak self] _ in
                guard let self = self, let viewController = self.viewController else { return }
                self.compressFile(viewController)
            }))

            // Add compress/extract options for archives
            let fileExtension = file.url.pathExtension.lowercased()
            if ["zip", "rar", "tar", "gz", "7z"].contains(fileExtension) {
                actions.append(UIAction(title: "Extract", image: UIImage(systemName: "archivebox.fill"), handler: { [weak self] _ in
                    guard let self = self, let viewController = self.viewController as? HomeViewController else { return }
                    viewController.extractArchive(self.file)
                }))
            }

            // Add edit option for text files
            if ["txt", "md", "swift", "h", "m", "c", "cpp", "js", "html", "css", "json", "strings", "plist"].contains(fileExtension) {
                actions.append(UIAction(title: "Edit", image: UIImage(systemName: "pencil.line"), handler: { [weak self] _ in
                    guard let self = self, let viewController = self.viewController else { return }
                    self.editFile(viewController)
                }))
            }

            // Add sign option for IPA files
            if fileExtension == "ipa" {
                actions.append(UIAction(title: "Sign IPA", image: UIImage(systemName: "signature"), handler: { [weak self] _ in
                    guard let self = self, let viewController = self.viewController else { return }
                    self.signIPA(viewController)
                }))
            }
        }

        // Delete action - always available but in destructive section
        let deleteAction = UIAction(title: "Delete", image: UIImage(systemName: "trash"), attributes: .destructive, handler: { [weak self] _ in
            guard let self = self, let viewController = self.viewController as? HomeViewController else { return }

            // Find index of file in the file list
            if let index = viewController.fileList.firstIndex(where: { $0.url == self.file.url }) {
                viewController.deleteFile(at: index)
            }
        })

        // Create menu with actions
        return UIMenu(title: file.name, children: actions + [deleteAction])
    }

    // MARK: - Action Handlers

    /// Share a file using the system share sheet
    /// - Parameter viewController: The view controller to present from
    private func shareFile(_ viewController: UIViewController) {
        let activityViewController = UIActivityViewController(activityItems: [file.url], applicationActivities: nil)

        // For iPad support
        if let popoverController = activityViewController.popoverPresentationController {
            popoverController.sourceView = viewController.view
            popoverController.sourceRect = CGRect(x: viewController.view.bounds.midX, y: viewController.view.bounds.midY, width: 0, height: 0)
            popoverController.permittedArrowDirections = []
        }

        viewController.present(activityViewController, animated: true, completion: nil)
    }

    /// Rename a file
    /// - Parameter viewController: The view controller to present from
    private func renameFile(_: UIViewController) {
        // Using HomeViewController's rename method directly as it's implemented there
    }

    /// Compress a file
    /// - Parameter viewController: The view controller to present from
    private func compressFile(_ viewController: UIViewController) {
        let zipURL = file.url.deletingPathExtension().appendingPathExtension("zip")

        // Check if zip already exists
        if FileManager.default.fileExists(atPath: zipURL.path) {
            let alert = UIAlertController(
                title: "File Exists",
                message: "A zip file with this name already exists. Overwrite?",
                preferredStyle: .alert
            )

            let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)
            let overwriteAction = UIAlertAction(title: "Overwrite", style: .destructive) { [weak self] _ in
                guard let self = self else { return }
                self.performCompression(viewController, zipURL: zipURL)
            }

            alert.addAction(cancelAction)
            alert.addAction(overwriteAction)
            viewController.present(alert, animated: true)
        } else {
            performCompression(viewController, zipURL: zipURL)
        }
    }

    /// Compress a directory
    /// - Parameter viewController: The view controller to present from
    private func compressDirectory(_ viewController: UIViewController) {
        let zipURL = file.url.appendingPathExtension("zip")

        // Check if zip already exists
        if FileManager.default.fileExists(atPath: zipURL.path) {
            let alert = UIAlertController(
                title: "File Exists",
                message: "A zip file with this name already exists. Overwrite?",
                preferredStyle: .alert
            )

            let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)
            let overwriteAction = UIAlertAction(title: "Overwrite", style: .destructive) { [weak self] _ in
                guard let self = self else { return }
                self.performCompression(viewController, zipURL: zipURL)
            }

            alert.addAction(cancelAction)
            alert.addAction(overwriteAction)
            viewController.present(alert, animated: true)
        } else {
            performCompression(viewController, zipURL: zipURL)
        }
    }

    /// Perform the actual compression
    /// - Parameters:
    ///   - viewController: The view controller to present from
    ///   - zipURL: The URL of the zip file to create
    private func performCompression(_ viewController: UIViewController, zipURL: URL) {
        // Show activity indicator
        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.center = viewController.view.center
        activityIndicator.startAnimating()
        viewController.view.addSubview(activityIndicator)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                // Try to delete existing zip if it exists
                if FileManager.default.fileExists(atPath: zipURL.path) {
                    try FileManager.default.removeItem(at: zipURL)
                }

                // Create the zip file
                if self.file.isDirectory {
                    try FileManager.default.zipItem(at: self.file.url, to: zipURL)
                } else {
                    // For single files, we need to create a zip with just that file
                    try FileManager.default.zipItem(at: self.file.url, to: zipURL)
                }

                DispatchQueue.main.async {
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()

                    // Notify of successful compression
                    HapticFeedbackGenerator.generateNotificationFeedback(type: .success)
                    self.onActionPerformed?()

                    let successAlert = UIAlertController(
                        title: "Compression Complete",
                        message: "File compressed successfully.",
                        preferredStyle: .alert
                    )
                    successAlert.addAction(UIAlertAction(title: "OK", style: .default))
                    viewController.present(successAlert, animated: true)
                }
            } catch {
                DispatchQueue.main.async {
                    activityIndicator.stopAnimating()
                    activityIndicator.removeFromSuperview()

                    Debug.shared.log(message: "Compression error: \(error.localizedDescription)", type: .error)

                    let errorAlert = UIAlertController(
                        title: "Compression Failed",
                        message: error.localizedDescription,
                        preferredStyle: .alert
                    )
                    errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
                    viewController.present(errorAlert, animated: true)
                }
            }
        }
    }

    /// Edit a text file
    /// - Parameter viewController: The view controller to present from
    private func editFile(_ viewController: UIViewController) {
        let fileExtension = file.url.pathExtension.lowercased()

        if fileExtension == "plist" {
            let editor = PlistEditorViewController(fileURL: file.url)
            viewController.navigationController?.pushViewController(editor, animated: true)
        } else {
            let editor = TextEditorViewController(fileURL: file.url)
            viewController.navigationController?.pushViewController(editor, animated: true)
        }
    }

    /// Sign an IPA file
    /// - Parameter viewController: The view controller to present from
    private func signIPA(_ viewController: UIViewController) {
        // Open the IPA editor
        let editor = IPAEditorViewController(fileURL: file.url)
        viewController.navigationController?.pushViewController(editor, animated: true)
    }
}

/// Extension to add long press context menu to table view cells
extension UITableViewCell {
    /// Add a context menu to this cell for a file
    /// - Parameters:
    ///   - file: The file to create a menu for
    ///   - viewController: The view controller that owns this menu
    ///   - onActionPerformed: Callback for when a file action is performed
    func addContextMenu(for file: File, in viewController: UIViewController, onActionPerformed: @escaping () -> Void) {
        // Remove any existing interactions to avoid duplicates
        for interaction in interactions {
            if interaction is UIContextMenuInteraction {
                removeInteraction(interaction)
            }
        }

        // Create the context menu handler
        let contextMenuHandler = FileContextMenu(for: file, in: viewController)
        contextMenuHandler.onActionPerformed = onActionPerformed

        // Create and add the interaction
        let interaction = UIContextMenuInteraction(delegate: contextMenuHandler)
        addInteraction(interaction)

        // Store the context menu handler to keep it alive
        objc_setAssociatedObject(self, "fileContextMenuKey", contextMenuHandler, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
}
