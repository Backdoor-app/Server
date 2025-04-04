// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import ZIPFoundation

class IPAEditorViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    // MARK: - Properties

    /// The URL of the IPA file being examined
    private let fileURL: URL

    /// Table view for displaying extracted contents
    private let tableView: UITableView

    /// Toolbar for actions
    private let toolbar: UIToolbar

    /// Activity indicator for long operations
    private let activityIndicator: UIActivityIndicatorView

    /// List of extracted file URLs
    private var contents: [URL] = []

    /// Current directory being displayed
    private var currentDirectory: URL?

    /// Stack of previous directories for navigation
    private var directoryStack: [URL] = []

    /// Temporary directory where IPA is extracted
    private var tempDirectory: URL?

    /// File manager instance
    private let fileManager = FileManager.default

    // MARK: - Initialization

    /// Initialize with the URL of an IPA file
    /// - Parameter fileURL: The URL of the IPA file to examine
    init(fileURL: URL) {
        self.fileURL = fileURL
        self.tableView = UITableView()
        self.toolbar = UIToolbar()
        self.activityIndicator = UIActivityIndicatorView(style: .large)
        super.init(nibName: nil, bundle: nil)
        title = fileURL.lastPathComponent
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        extractIPAContents()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cleanupTempDirectory()
    }

    deinit {
        cleanupTempDirectory()
    }

    // MARK: - UI Setup

    /// Sets up the user interface
    private func setupUI() {
        view.backgroundColor = UIColor.systemBackground.withAlphaComponent(0.95)
        view.layer.applyFuturisticShadow()

        setupTableView()
        setupToolbar()
        setupActivityIndicator()
        setupConstraints()
    }

    /// Sets up the table view
    private func setupTableView() {
        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "Cell")
        tableView.layer.cornerRadius = 10
        tableView.layer.borderColor = UIColor.systemCyan.withAlphaComponent(0.2).cgColor
        tableView.layer.borderWidth = 1
        view.addSubview(tableView)

        tableView.isAccessibilityElement = true
        tableView.accessibilityLabel = "IPA Contents"
    }

    /// Sets up the toolbar
    private func setupToolbar() {
        toolbar.translatesAutoresizingMaskIntoConstraints = false

        let backButton = UIBarButtonItem(
            image: UIImage(systemName: "chevron.backward"),
            style: .plain,
            target: self,
            action: #selector(navigateBack)
        )
        backButton.isEnabled = false

        let infoButton = UIBarButtonItem(
            image: UIImage(systemName: "info.circle"),
            style: .plain,
            target: self,
            action: #selector(showIPAInfo)
        )

        let refreshButton = UIBarButtonItem(
            barButtonSystemItem: .refresh,
            target: self,
            action: #selector(reloadContents)
        )

        let flexibleSpace = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)

        toolbar.items = [backButton, flexibleSpace, infoButton, refreshButton]
        toolbar.tintColor = .systemCyan
        toolbar.layer.cornerRadius = 10
        view.addSubview(toolbar)
    }

    /// Sets up the activity indicator
    private func setupActivityIndicator() {
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.hidesWhenStopped = true
        activityIndicator.color = .systemCyan
        view.addSubview(activityIndicator)
    }

    /// Sets up layout constraints
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            tableView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            tableView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            tableView.bottomAnchor.constraint(equalTo: toolbar.topAnchor, constant: -10),

            toolbar.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            toolbar.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            toolbar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10),
            toolbar.heightAnchor.constraint(equalToConstant: 44),

            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])
    }

    // MARK: - IPA Processing

    /// Extract the contents of the IPA file
    private func extractIPAContents() {
        activityIndicator.startAnimating()

        // Create a unique temporary directory
        let newTempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        self.tempDirectory = newTempDirectory

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                // Create temp directory
                try self.fileManager.createDirectory(at: newTempDirectory, withIntermediateDirectories: true, attributes: nil)

                // Unzip IPA to temp directory
                try self.fileManager.unzipItem(at: self.fileURL, to: newTempDirectory)

                // Get the top-level contents
                let extractedContents = try self.fileManager.contentsOfDirectory(
                    at: newTempDirectory,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles
                )

                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.contents = extractedContents
                    self.currentDirectory = newTempDirectory
                    self.directoryStack = []
                    self.tableView.reloadData()
                    self.activityIndicator.stopAnimating()

                    // Update back button state
                    if let backButton = self.toolbar.items?.first {
                        backButton.isEnabled = !self.directoryStack.isEmpty
                    }

                    // Install a cleanup handler for app termination
                    NotificationCenter.default.addObserver(
                        self,
                        selector: #selector(self.cleanupTempDirectory),
                        name: UIApplication.willTerminateNotification,
                        object: nil
                    )
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.activityIndicator.stopAnimating()
                    Debug.shared.log(message: "IPA extraction error: \(error.localizedDescription)", type: .error)
                    self.presentAlert(title: "Error", message: "Failed to extract IPA: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Navigate into a subdirectory
    /// - Parameter directoryURL: The URL of the directory to navigate into
    private func navigateToDirectory(_ directoryURL: URL) {
        activityIndicator.startAnimating()

        // Save current directory to stack
        if let currentDir = currentDirectory {
            directoryStack.append(currentDir)
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                // Get contents of the selected directory
                let directoryContents = try self.fileManager.contentsOfDirectory(
                    at: directoryURL,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles
                )

                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.contents = directoryContents
                    self.currentDirectory = directoryURL
                    self.tableView.reloadData()
                    self.activityIndicator.stopAnimating()

                    // Update back button state
                    if let backButton = self.toolbar.items?.first {
                        backButton.isEnabled = !self.directoryStack.isEmpty
                    }

                    // Update navigation title
                    self.title = directoryURL.lastPathComponent
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.activityIndicator.stopAnimating()
                    Debug.shared.log(message: "Directory navigation error: \(error.localizedDescription)", type: .error)
                    self.presentAlert(title: "Error", message: "Failed to read directory: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Navigate back to the previous directory
    @objc private func navigateBack() {
        guard !directoryStack.isEmpty else { return }

        let previousDirectory = directoryStack.removeLast()
        navigateToDirectory(previousDirectory)

        // We need to remove it again since navigateToDirectory will add it back
        if !directoryStack.isEmpty {
            directoryStack.removeLast()
        }
    }

    /// Reload the current directory contents
    @objc private func reloadContents() {
        guard let currentDir = currentDirectory else { return }

        activityIndicator.startAnimating()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                let directoryContents = try self.fileManager.contentsOfDirectory(
                    at: currentDir,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles
                )

                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.contents = directoryContents
                    self.tableView.reloadData()
                    self.activityIndicator.stopAnimating()

                    HapticFeedbackGenerator.generateHapticFeedback(style: .light)
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.activityIndicator.stopAnimating()
                    Debug.shared.log(message: "Directory reload error: \(error.localizedDescription)", type: .error)
                    self.presentAlert(title: "Error", message: "Failed to reload directory: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Remove temporary directory when done
    @objc private func cleanupTempDirectory() {
        // Remove observer
        NotificationCenter.default.removeObserver(self, name: UIApplication.willTerminateNotification, object: nil)

        // Clean up temp directory
        if let tempDir = tempDirectory, fileManager.fileExists(atPath: tempDir.path) {
            do {
                try fileManager.removeItem(at: tempDir)
                Debug.shared.log(message: "Cleaned up IPA temp directory: \(tempDir.path)", type: .info)
            } catch {
                Debug.shared.log(message: "Failed to clean up temp directory: \(error.localizedDescription)", type: .error)
            }
        }
    }

    /// Show information about the IPA file
    @objc private func showIPAInfo() {
        activityIndicator.startAnimating()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                // Get file attributes
                let fileAttributes = try self.fileManager.attributesOfItem(atPath: self.fileURL.path)
                let infoPlistURL = findInfoPlist()

                var appInfo: [String: Any] = [:]
                if let infoPlistURL = infoPlistURL {
                    // Parse Info.plist if found
                    let infoPlistData = try Data(contentsOf: infoPlistURL)
                    if let plistDict = try PropertyListSerialization.propertyList(from: infoPlistData, options: [], format: nil) as? [String: Any] {
                        appInfo = plistDict
                    }
                }

                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.activityIndicator.stopAnimating()
                    self.showIPAInfoAlert(fileAttributes: fileAttributes, appInfo: appInfo)
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.activityIndicator.stopAnimating()
                    Debug.shared.log(message: "IPA info error: \(error.localizedDescription)", type: .error)
                    self.presentAlert(title: "Error", message: "Failed to get IPA info: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Find the Info.plist file in the IPA
    /// - Returns: URL of the Info.plist if found, nil otherwise
    private func findInfoPlist() -> URL? {
        guard let tempDir = tempDirectory else { return nil }

        // Typical Info.plist paths in IPA
        let possiblePaths = [
            "Payload/*.app/Info.plist",
            "*.app/Info.plist",
        ]

        for pattern in possiblePaths {
            do {
                // First get the app directories
                let components = pattern.components(separatedBy: "/")
                guard components.count >= 1 else { continue }

                let appDirPattern = components[0]
                var appDirs: [URL] = []

                if appDirPattern == "Payload" {
                    // Special case for Payload directory which is standard in IPAs
                    let payloadDir = tempDir.appendingPathComponent("Payload")
                    if fileManager.fileExists(atPath: payloadDir.path) {
                        appDirs = (try? fileManager.contentsOfDirectory(at: payloadDir, includingPropertiesForKeys: nil)) ?? []
                    }
                } else {
                    // Search for directories matching the pattern
                    let contents = try fileManager.contentsOfDirectory(at: tempDir, includingPropertiesForKeys: nil)
                    appDirs = contents.filter { $0.pathExtension == "app" }
                }

                // Now look for Info.plist in each app directory
                for appDir in appDirs {
                    let infoPlist = appDir.appendingPathComponent("Info.plist")
                    if fileManager.fileExists(atPath: infoPlist.path) {
                        return infoPlist
                    }
                }
            } catch {
                Debug.shared.log(message: "Error searching for Info.plist: \(error.localizedDescription)", type: .error)
            }
        }

        return nil
    }

    /// Display an alert with IPA information
    /// - Parameters:
    ///   - fileAttributes: File attributes of the IPA
    ///   - appInfo: Contents of Info.plist if available
    private func showIPAInfoAlert(fileAttributes: [FileAttributeKey: Any], appInfo: [String: Any]) {
        var message = ""

        // File info
        if let fileSize = fileAttributes[.size] as? UInt64 {
            message += "Size: \(ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file))\n"
        }

        if let creationDate = fileAttributes[.creationDate] as? Date {
            let dateFormatter = DateFormatter()
            dateFormatter.dateStyle = .medium
            dateFormatter.timeStyle = .short
            message += "Created: \(dateFormatter.string(from: creationDate))\n"
        }

        // App info
        if !appInfo.isEmpty {
            message += "\nApplication Info:\n"

            if let bundleName = appInfo["CFBundleName"] as? String {
                message += "Name: \(bundleName)\n"
            }

            if let bundleID = appInfo["CFBundleIdentifier"] as? String {
                message += "Bundle ID: \(bundleID)\n"
            }

            if let version = appInfo["CFBundleShortVersionString"] as? String {
                message += "Version: \(version)\n"
            }

            if let buildNumber = appInfo["CFBundleVersion"] as? String {
                message += "Build: \(buildNumber)\n"
            }

            if let minOSVersion = appInfo["MinimumOSVersion"] as? String {
                message += "Min iOS: \(minOSVersion)\n"
            }
        } else {
            message += "\nNo application info found in IPA."
        }

        presentAlert(title: "IPA Information", message: message)
    }

    // MARK: - Actions

    /// Opens a file using appropriate editor based on file type
    /// - Parameter fileURL: The URL of the file to open
    private func openFile(_ fileURL: URL) {
        let fileExtension = fileURL.pathExtension.lowercased()

        switch fileExtension {
            case "plist":
                let editor = PlistEditorViewController(fileURL: fileURL)
                navigationController?.pushViewController(editor, animated: true)

            case "txt", "strings", "h", "m", "swift", "c", "cpp", "md", "json", "xml", "html", "css", "js":
                let editor = TextEditorViewController(fileURL: fileURL)
                navigationController?.pushViewController(editor, animated: true)

            default:
                // For binary or unknown files, use hex editor
                let isDirectory = (try? fileURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false

                if isDirectory {
                    navigateToDirectory(fileURL)
                } else {
                    // Ask user which editor to use
                    let alert = UIAlertController(
                        title: "Open File",
                        message: "How would you like to open this file?",
                        preferredStyle: .actionSheet
                    )

                    let hexEditorAction = UIAlertAction(title: "Hex Editor", style: .default) { [weak self] _ in
                        guard let self = self else { return }
                        let editor = HexEditorViewController(fileURL: fileURL)
                        self.navigationController?.pushViewController(editor, animated: true)
                    }

                    let textEditorAction = UIAlertAction(title: "Text Editor", style: .default) { [weak self] _ in
                        guard let self = self else { return }
                        let editor = TextEditorViewController(fileURL: fileURL)
                        self.navigationController?.pushViewController(editor, animated: true)
                    }

                    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)

                    alert.addAction(hexEditorAction)
                    alert.addAction(textEditorAction)
                    alert.addAction(cancelAction)

                    present(alert, animated: true, completion: nil)
                }
        }
    }

    /// Present an alert with a title and message
    /// - Parameters:
    ///   - title: Alert title
    ///   - message: Alert message
    private func presentAlert(title: String, message: String) {
        let alert = UIAlertController(
            title: title,
            message: message,
            preferredStyle: .alert
        )

        let okAction = UIAlertAction(title: "OK", style: .default)
        alert.addAction(okAction)

        present(alert, animated: true, completion: nil)
    }

    // MARK: - UITableViewDataSource

    func tableView(_: UITableView, numberOfRowsInSection _: Int) -> Int {
        return contents.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath)
        let fileURL = contents[indexPath.row]

        // Configure cell
        cell.textLabel?.text = fileURL.lastPathComponent

        // Determine if it's a directory
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: fileURL.path, isDirectory: &isDirectory) && isDirectory.boolValue {
            cell.imageView?.image = UIImage(systemName: "folder")
            cell.accessoryType = .disclosureIndicator
        } else {
            // Set icon based on file type
            let fileExtension = fileURL.pathExtension.lowercased()
            switch fileExtension {
                case "plist":
                    cell.imageView?.image = UIImage(systemName: "doc.text")
                case "txt", "strings", "h", "m", "swift", "c", "cpp":
                    cell.imageView?.image = UIImage(systemName: "doc.plaintext")
                case "png", "jpg", "jpeg", "gif":
                    cell.imageView?.image = UIImage(systemName: "photo")
                default:
                    cell.imageView?.image = UIImage(systemName: "doc")
            }
            cell.accessoryType = .detailDisclosureButton
        }

        // Customize appearance
        cell.imageView?.tintColor = .systemCyan
        cell.backgroundColor = .clear

        return cell
    }

    // MARK: - UITableViewDelegate

    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let selectedURL = contents[indexPath.row]

        // Check if it's a directory
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: selectedURL.path, isDirectory: &isDirectory) && isDirectory.boolValue {
            // Navigate into the directory
            navigateToDirectory(selectedURL)
        } else {
            // Open the file with appropriate editor
            openFile(selectedURL)
        }

        tableView.deselectRow(at: indexPath, animated: true)
    }

    func tableView(_: UITableView, accessoryButtonTappedForRowWith indexPath: IndexPath) {
        let fileURL = contents[indexPath.row]

        // Show file info
        do {
            let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
            var infoText = "Name: \(fileURL.lastPathComponent)\n"

            if let fileSize = attributes[.size] as? UInt64 {
                infoText += "Size: \(ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file))\n"
            }

            if let creationDate = attributes[.creationDate] as? Date {
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                dateFormatter.timeStyle = .short
                infoText += "Created: \(dateFormatter.string(from: creationDate))\n"
            }

            presentAlert(title: "File Information", message: infoText)
        } catch {
            presentAlert(title: "Error", message: "Could not get file information: \(error.localizedDescription)")
        }
    }
}
