// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import QuickLook
import UIKit

// MARK: - Extensions to provide all-in-one comprehensive file management

extension DirectoryViewController {
    /// Set up the file management UI elements
    func setupFileManagementUI() {
        // Add edit button for quick file creation
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .add,
            target: self,
            action: #selector(showAddFileOptions)
        )
    }

    @objc private func showAddFileOptions() {
        let alertController = UIAlertController(
            title: "Add New",
            message: "What would you like to create?",
            preferredStyle: .actionSheet
        )

        // New folder option
        let folderAction = UIAlertAction(title: "New Folder", style: .default) { [weak self] _ in
            guard let self = self else { return }
            // Create a File object for the current directory
            let currentDirFile = File(url: self.documentsDirectory)
            self.createNewFolder(in: currentDirFile)
        }
        folderAction.setValue(UIImage(systemName: "folder.badge.plus"), forKey: "image")

        // New text file option
        let textFileAction = UIAlertAction(title: "New Text File", style: .default) { [weak self] _ in
            guard let self = self else { return }
            // Create a File object for the current directory
            let currentDirFile = File(url: self.documentsDirectory)
            self.createNewFile(in: currentDirFile)
        }
        textFileAction.setValue(UIImage(systemName: "doc.badge.plus"), forKey: "image")

        // Import file option
        let importAction = UIAlertAction(title: "Import File", style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.importFile()
        }
        importAction.setValue(UIImage(systemName: "square.and.arrow.down"), forKey: "image")

        // Camera option (for taking photos)
        let cameraAction = UIAlertAction(title: "Take Photo", style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.takePhoto()
        }
        cameraAction.setValue(UIImage(systemName: "camera"), forKey: "image")

        // Cancel option
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)

        alertController.addAction(folderAction)
        alertController.addAction(textFileAction)
        alertController.addAction(importAction)
        alertController.addAction(cameraAction)
        alertController.addAction(cancelAction)

        // Support iPad
        if let popover = alertController.popoverPresentationController {
            popover.barButtonItem = navigationItem.rightBarButtonItem
        }

        present(alertController, animated: true, completion: nil)
    }

    /// Take a photo and save it to the current directory
    private func takePhoto() {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePicker = UIImagePickerController()
            imagePicker.sourceType = .camera
            imagePicker.delegate = self
            imagePicker.allowsEditing = true
            present(imagePicker, animated: true, completion: nil)
        } else {
            // Camera not available
            let alert = UIAlertController(
                title: "Camera Not Available",
                message: "This device does not have a camera or camera access is restricted.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true, completion: nil)
        }
    }
}

// MARK: - UIImagePickerControllerDelegate

extension DirectoryViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        picker.dismiss(animated: true)

        guard let image = info[.editedImage] as? UIImage ?? info[.originalImage] as? UIImage else {
            // No image selected
            return
        }

        // Save the image to the documents directory
        let timestamp = Date().timeIntervalSince1970
        let imageName = "IMG_\(Int(timestamp)).jpg"
        let imageURL = documentsDirectory.appendingPathComponent(imageName)

        // Show activity indicator
        activityIndicator.startAnimating()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            if let imageData = image.jpegData(compressionQuality: 0.8) {
                do {
                    try imageData.write(to: imageURL)

                    DispatchQueue.main.async {
                        self.activityIndicator.stopAnimating()
                        self.loadFiles()
                        HapticFeedbackGenerator.generateNotificationFeedback(type: .success)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.activityIndicator.stopAnimating()
                        self.utilities.handleError(
                            in: self,
                            error: error,
                            withTitle: "Image Save Error"
                        )
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    self.utilities.handleError(
                        in: self,
                        error: NSError(domain: "Image processing failed", code: 1, userInfo: nil),
                        withTitle: "Image Save Error"
                    )
                }
            }
        }
    }
}
