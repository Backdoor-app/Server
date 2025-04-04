// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class FileTableViewCell: UITableViewCell {
    let fileIconImageView = UIImageView()
    let fileNameLabel = UILabel()
    let fileSizeLabel = UILabel()
    let fileDateLabel = UILabel()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupUI()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupUI() {
        contentView.addSubview(fileIconImageView)
        contentView.addSubview(fileNameLabel)
        contentView.addSubview(fileSizeLabel)
        contentView.addSubview(fileDateLabel)

        fileIconImageView.translatesAutoresizingMaskIntoConstraints = false
        fileNameLabel.translatesAutoresizingMaskIntoConstraints = false
        fileSizeLabel.translatesAutoresizingMaskIntoConstraints = false
        fileDateLabel.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            fileIconImageView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            fileIconImageView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            fileIconImageView.widthAnchor.constraint(equalToConstant: 40),
            fileIconImageView.heightAnchor.constraint(equalToConstant: 40),

            fileNameLabel.leadingAnchor.constraint(equalTo: fileIconImageView.trailingAnchor, constant: 16),
            fileNameLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            fileNameLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),

            fileSizeLabel.leadingAnchor.constraint(equalTo: fileNameLabel.leadingAnchor),
            fileSizeLabel.topAnchor.constraint(equalTo: fileNameLabel.bottomAnchor, constant: 4),

            fileDateLabel.leadingAnchor.constraint(equalTo: fileNameLabel.leadingAnchor),
            fileDateLabel.topAnchor.constraint(equalTo: fileSizeLabel.bottomAnchor, constant: 4),
            fileDateLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8),
        ])

        fileIconImageView.contentMode = .scaleAspectFit
        fileIconImageView.tintColor = .systemCyan
        fileNameLabel.font = .systemFont(ofSize: 16, weight: .medium)
        fileSizeLabel.font = .systemFont(ofSize: 12, weight: .light)
        fileDateLabel.font = .systemFont(ofSize: 12, weight: .light)

        fileIconImageView.isAccessibilityElement = true
        fileIconImageView.accessibilityLabel = "File Icon"
        fileNameLabel.isAccessibilityElement = true
        fileNameLabel.accessibilityLabel = "File Name"
        fileSizeLabel.isAccessibilityElement = true
        fileSizeLabel.accessibilityLabel = "File Size"
        fileDateLabel.isAccessibilityElement = true
        fileDateLabel.accessibilityLabel = "File Date"
    }

    func configure(with file: File, in viewController: UIViewController, onActionPerformed: @escaping () -> Void) {
        fileNameLabel.text = file.name
        fileSizeLabel.text = ByteCountFormatter.string(fromByteCount: Int64(file.size), countStyle: .file)
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        fileDateLabel.text = dateFormatter.string(from: file.date)
        fileIconImageView.image = UIImage(named: file.iconName) ?? UIImage(systemName: file.iconName)
        accessibilityElements = [fileIconImageView, fileNameLabel, fileSizeLabel, fileDateLabel]

        // Add context menu for long press
        addContextMenu(for: file, in: viewController, onActionPerformed: onActionPerformed)

        // Setup long press gesture if it doesn't exist
        if gestureRecognizers?.contains(where: { $0 is UILongPressGestureRecognizer }) != true {
            let longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
            longPressGesture.minimumPressDuration = 0.5
            addGestureRecognizer(longPressGesture)
        }
    }

    @objc private func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            // This will trigger the context menu
            HapticFeedbackGenerator.generateHapticFeedback(style: .medium)
        }
    }
}

/// Represents a file in the file system with cached attributes for performance
class File: Equatable {
    /// URL of the file
    let url: URL

    /// File name (cached)
    let name: String

    /// File size in bytes (cached)
    let size: UInt64

    /// File modification date (cached)
    let date: Date

    /// Icon name for display (cached)
    let iconName: String

    /// Whether this represents a directory (cached)
    let isDirectory: Bool

    /// Initialize with a URL and load attributes
    /// - Parameter url: URL of the file
    init(url: URL) {
        self.url = url
        self.name = url.lastPathComponent

        // Get file attributes once to avoid repeated filesystem access
        let fileManager = FileManager.default

        // Load file attributes
        if let attributes = try? fileManager.attributesOfItem(atPath: url.path) {
            self.size = attributes[.size] as? UInt64 ?? 0
            self.date = attributes[.modificationDate] as? Date ?? Date.distantPast
            self.isDirectory = (attributes[.type] as? String) == "NSFileTypeDirectory"
        } else {
            // Default values if attributes can't be read
            self.size = 0
            self.date = Date.distantPast
            self.isDirectory = false
            Debug.shared.log(message: "Failed to get attributes for file: \(url.path)", type: .warning)
        }

        // Determine icon based on extension
        let extensionString = url.pathExtension.lowercased()
        if isDirectory {
            self.iconName = "folder"
        } else {
            switch extensionString {
                case "txt", "md", "strings", "json":
                    self.iconName = "iconText"
                case "plist", "entitlements":
                    self.iconName = "iconPlist"
                case "ipa":
                    self.iconName = "iconIPA"
                case "zip", "gz", "tar", "7z":
                    self.iconName = "iconZip"
                case "pdf":
                    self.iconName = "iconPDF"
                case "png", "jpg", "jpeg", "gif", "heic", "webp":
                    self.iconName = "iconImage"
                case "mp3", "m4a", "wav", "aac":
                    self.iconName = "iconAudio"
                case "mp4", "mov", "m4v", "3gp", "avi", "flv", "mpg", "wmv", "mkv":
                    self.iconName = "iconVideo"
                case "swift", "h", "m", "c", "cpp", "js", "html", "css", "py", "java", "xml", "json":
                    self.iconName = "iconCode"
                case "doc", "docx":
                    self.iconName = "doc.text"
                case "xls", "xlsx", "numbers":
                    self.iconName = "doc.text.fill"
                case "ppt", "pptx", "key":
                    self.iconName = "chart.bar.doc.horizontal"
                case "pages":
                    self.iconName = "doc"
                default:
                    self.iconName = "iconGeneric"
            }
        }
    }

    /// Format the file size for display
    /// - Returns: Human-readable file size string
    func formattedSize() -> String {
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }

    /// Format the date for display
    /// - Returns: Formatted date string
    func formattedDate() -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    /// Equatable implementation
    static func == (lhs: File, rhs: File) -> Bool {
        return lhs.url == rhs.url
    }
}
