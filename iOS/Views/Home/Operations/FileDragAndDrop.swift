// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import MobileCoreServices
import UIKit
import UniformTypeIdentifiers

extension HomeViewController {
    // MARK: - UITableViewDragDelegate

    func tableView(_: UITableView, itemsForBeginning session: UIDragSession, at indexPath: IndexPath) -> [UIDragItem] {
        // Get the file at the index path
        let file = searchController.isActive ? filteredFileList[indexPath.row] : fileList[indexPath.row]

        // Check if file exists
        guard FileManager.default.fileExists(atPath: file.url.path) else {
            return []
        }

        // Create a drag item with the file URL
        let itemProvider = NSItemProvider(object: file.url as NSURL)
        let dragItem = UIDragItem(itemProvider: itemProvider)
        dragItem.localObject = file.name
        
        // Store the file name in the session's localContext for later use
        session.localContext = file.name

        return [dragItem]
    }

    // MARK: - UITableViewDropDelegate

    func tableView(_ tableView: UITableView, performDropWith coordinator: UITableViewDropCoordinator) {
        // Handle internal reordering
        if coordinator.session.hasItemsConforming(toTypeIdentifiers: [UTType.url.identifier]) {
            tableHandlers.tableView(
                tableView,
                performDropWith: coordinator,
                fileList: &fileList,
                documentsDirectory: documentsDirectory,
                loadFiles: loadFiles
            )
        } else {
            // Handle external drops
            for item in coordinator.items {
                handleExternalDrop(item, coordinator: coordinator)
            }
        }
    }

    func handleExternalDrop(_ dropItem: UITableViewDropItem, coordinator _: UITableViewDropCoordinator) {
        // Check for URLs
        if dropItem.dragItem.itemProvider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
            dropItem.dragItem.itemProvider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { [weak self] urlData, _ in
                guard let self = self, let urlData = urlData else { return }

                DispatchQueue.main.async {
                    // Show loading indicator
                    self.activityIndicator.startAnimating()

                    if let url = urlData as? URL {
                        // Process the dropped URL
                        self.handleImportedFile(url: url)
                    } else if let urlString = urlData as? String, let url = URL(string: urlString) {
                        // Process the dropped URL string
                        self.handleImportedFile(url: url)
                    } else {
                        self.activityIndicator.stopAnimating()
                    }
                }
            }
        }

        // Check for images
        if dropItem.dragItem.itemProvider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            dropItem.dragItem.itemProvider.loadItem(forTypeIdentifier: UTType.image.identifier, options: nil) { [weak self] imageData, _ in
                guard let self = self, let imageData = imageData else { return }

                DispatchQueue.main.async {
                    // Show loading indicator
                    self.activityIndicator.startAnimating()

                    if let url = imageData as? URL {
                        // Process the dropped image URL
                        self.handleImportedFile(url: url)
                    } else if let image = imageData as? UIImage, let data = image.pngData() {
                        // Save the image to a temporary file
                        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("dropped_image_\(UUID().uuidString).png")
                        do {
                            try data.write(to: tempURL)
                            self.handleImportedFile(url: tempURL)
                        } catch {
                            self.activityIndicator.stopAnimating()
                            self.utilities.handleError(in: self, error: error, withTitle: "Image Import Error")
                        }
                    } else {
                        self.activityIndicator.stopAnimating()
                    }
                }
            }
        }

        // Check for text
        if dropItem.dragItem.itemProvider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
            dropItem.dragItem.itemProvider.loadItem(forTypeIdentifier: UTType.plainText.identifier, options: nil) { [weak self] textData, _ in
                guard let self = self, let text = textData as? String else { return }

                DispatchQueue.main.async {
                    // Show loading indicator
                    self.activityIndicator.startAnimating()

                    // Create a text file with the dropped text
                    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("dropped_text_\(UUID().uuidString).txt")
                    do {
                        try text.write(to: tempURL, atomically: true, encoding: .utf8)
                        self.handleImportedFile(url: tempURL)
                    } catch {
                        self.activityIndicator.stopAnimating()
                        self.utilities.handleError(in: self, error: error, withTitle: "Text Import Error")
                    }
                }
            }
        }
    }

    func tableView(_: UITableView, dropSessionDidUpdate session: UIDropSession, withDestinationIndexPath _: IndexPath?) -> UITableViewDropProposal {
        // Determine if the drop is within the app or from outside
        if session.localDragSession != nil {
            // Internal reordering
            return UITableViewDropProposal(operation: .move, intent: .insertAtDestinationIndexPath)
        } else {
            // External drop
            return UITableViewDropProposal(operation: .copy, intent: .insertAtDestinationIndexPath)
        }
    }
}
