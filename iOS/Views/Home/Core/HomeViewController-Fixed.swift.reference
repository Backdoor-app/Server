// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import ZIPFoundation

class HomeViewController: UIViewController, UISearchResultsUpdating, UIDocumentPickerDelegate, FileHandlingDelegate, UITableViewDelegate, UITableViewDataSource, UITableViewDragDelegate, UITableViewDropDelegate {
    // MARK: - Properties

    var fileList: [File] = []
    // Changed from private to internal to allow access from extensions
    var filteredFileList: [File] = []
    let fileManager = FileManager.default
    let searchController = UISearchController(searchResultsController: nil)
    var sortOrder: SortOrder = .name
    let fileHandlers = HomeViewFileHandlers()
    let utilities = HomeViewUtilities()
    let tableHandlers = HomeViewTableHandlers(utilities: HomeViewUtilities()) // Initialize with utilities

    /// The base directory for storing files
    /// Uses the app's documents directory with a "files" subdirectory
    var documentsDirectory: URL {
        // Get the documents directory safely
        guard let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            // This is a serious error - log it and return a fallback directory
            Debug.shared.log(message: "Failed to get documents directory, using temporary directory as fallback", type: .error)
            return FileManager.default.temporaryDirectory.appendingPathComponent("files")
        }

        // Create the files subdirectory
        let directory = documentsURL.appendingPathComponent("files")
        createFilesDirectoryIfNeeded(at: directory)
        return directory
    }

    enum SortOrder: String {
        case name, date, size
    }

    var activityIndicator: UIActivityIndicatorView {
        return HomeViewUI.activityIndicator
    }

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupActivityIndicator()
        loadFiles()
        configureTableView()
    }

    deinit {
        // No observation to invalidate
    }

    // MARK: - UI Setup

    private func setupUI() {
        view.backgroundColor = UIColor.systemBackground.withAlphaComponent(0.9)
        view.layer.applyFuturisticShadow()

        let navItem = UINavigationItem(title: "File Nexus")
        let menuButton = UIBarButtonItem(image: UIImage(systemName: "ellipsis.circle.fill"), style: .plain, target: self, action: #selector(showMenu))
        let uploadButton = UIBarButtonItem(customView: HomeViewUI.uploadButton)
        let addButton = UIBarButtonItem(image: UIImage(systemName: "folder.badge.plus"), style: .plain, target: self, action: #selector(addDirectory))

        HomeViewUI.uploadButton.addTarget(self, action: #selector(importFile), for: .touchUpInside)
        HomeViewUI.uploadButton.addGradientBackground()
        navItem.rightBarButtonItems = [menuButton, uploadButton, addButton]
        HomeViewUI.navigationBar.setItems([navItem], animated: false)
        view.addSubview(HomeViewUI.navigationBar)

        searchController.searchResultsUpdater = self
        searchController.obscuresBackgroundDuringPresentation = false
        searchController.searchBar.placeholder = "Search Files"
        searchController.searchBar.tintColor = .systemBlue
        navigationItem.searchController = searchController
        definesPresentationContext = true

        view.addSubview(HomeViewUI.fileListTableView)
        NSLayoutConstraint.activate([
            HomeViewUI.navigationBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            HomeViewUI.navigationBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            HomeViewUI.navigationBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            HomeViewUI.fileListTableView.topAnchor.constraint(equalTo: HomeViewUI.navigationBar.bottomAnchor, constant: 10),
            HomeViewUI.fileListTableView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            HomeViewUI.fileListTableView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            HomeViewUI.fileListTableView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10),
        ])

        // Apply animation effect
        applyFuturisticEffect()
    }

    /// Applies a futuristic transition effect to the view
    @objc public func applyFuturisticEffect() {
        // Create a snapshot of the current view
        guard let snapshot = view.snapshotView(afterScreenUpdates: false) else { return }
        view.addSubview(snapshot)

        // Apply a scale and fade animation
        UIView.animate(withDuration: 0.5, delay: 0, options: .curveEaseInOut, animations: {
            snapshot.alpha = 0
            snapshot.transform = CGAffineTransform(scaleX: 1.1, y: 1.1)
        }, completion: { _ in
            snapshot.removeFromSuperview()
        })
    }

    private func setupActivityIndicator() {
        view.addSubview(activityIndicator)
        NSLayoutConstraint.activate([
            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])
    }

    private func configureTableView() {
        HomeViewUI.fileListTableView.delegate = self
        HomeViewUI.fileListTableView.dataSource = self
        HomeViewUI.fileListTableView.dragDelegate = self
        HomeViewUI.fileListTableView.dropDelegate = self
        HomeViewUI.fileListTableView.register(FileTableViewCell.self, forCellReuseIdentifier: "FileCell")
        HomeViewUI.fileListTableView.backgroundColor = .clear
        HomeViewUI.fileListTableView.layer.cornerRadius = 15
        HomeViewUI.fileListTableView.layer.applyFuturisticShadow()
    }

    /// Creates the files directory if it doesn't exist
    /// - Parameter directory: The directory URL to create
    /// - Returns: True if the directory exists or was created successfully, false otherwise
    @discardableResult
    private func createFilesDirectoryIfNeeded(at directory: URL) -> Bool {
        // Check if directory already exists
        if fileManager.fileExists(atPath: directory.path) {
            return true
        }

        // Directory doesn't exist, try to create it
        do {
            try fileManager.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: nil
            )
            Debug.shared.log(message: "Created directory: \(directory.path)", type: .info)
            return true
        } catch {
            // Log the error and show to user
            Debug.shared.log(message: "Failed to create directory: \(error.localizedDescription)", type: .error)
            utilities.handleError(in: self, error: error, withTitle: "Directory Creation Error")
            return false
        }
    }

    private func saveState() {
        UserDefaults.standard.set(sortOrder.rawValue, forKey: "sortOrder")
    }

    // MARK: - File Operations

    /// Loads files from the documents directory and updates the UI
    func loadFiles() {
        // Start loading indicator
        activityIndicator.startAnimating()

        // Ensure the documents directory exists before trying to load files
        if !createFilesDirectoryIfNeeded(at: documentsDirectory) {
            // If we can't create the directory, stop loading
            activityIndicator.stopAnimating()
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Capture the start time for performance measurement
            let startTime = Date()

            do {
                // Load directory contents with necessary file attributes
                let fileURLs = try self.fileManager.contentsOfDirectory(
                    at: self.documentsDirectory,
                    includingPropertiesForKeys: [
                        .creationDateKey,
                        .contentModificationDateKey,
                        .fileSizeKey,
                        .isDirectoryKey,
                    ],
                    options: .skipsHiddenFiles
                )

                // Create File objects with cached attributes for better performance
                // This avoids accessing the filesystem repeatedly when displaying files
                var fileObjects: [File] = []
                for fileURL in fileURLs {
                    let file = File(url: fileURL)
                    fileObjects.append(file)
                }

                // Calculate loading time for performance monitoring
                let loadTime = Date().timeIntervalSince(startTime)
                Debug.shared.log(message: "Loaded \(fileObjects.count) files in \(String(format: "%.3f", loadTime))s", type: .info)

                // Update UI on main thread
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.fileList = fileObjects
                    self.sortFiles()
                    HomeViewUI.fileListTableView.reloadData()
                    self.activityIndicator.stopAnimating()

                    // If no files, show a helpful message
                    if fileObjects.isEmpty {
                        self.showEmptyStateMessage()
                    } else {
                        self.hideEmptyStateMessage()
                    }
                }
            } catch {
                Debug.shared.log(message: "Failed to load files: \(error.localizedDescription)", type: .error)

                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }

                    self.activityIndicator.stopAnimating()
                    self.utilities.handleError(in: self, error: error, withTitle: "File Load Error")

                    // Show empty state with error
                    self.showEmptyStateMessage(withError: error)
                }
            }
        }
    }

    // MARK: - UI Actions
    
    /// Creates a new directory
    @objc func addDirectory() {
        let alertController = UIAlertController(title: "Add Directory", message: "Enter the name of the new directory", preferredStyle: .alert)
        alertController.addTextField { textField in
            textField.placeholder = "Directory Name"
            textField.autocapitalizationType = .none
        }

        let createAction = UIAlertAction(title: "Create", style: .default) { [weak self] _ in
            guard let self = self,
                  let textField = alertController.textFields?.first,
                  let directoryName = textField.text?.trimmingCharacters(in: .whitespaces),
                  !directoryName.isEmpty else { return }

            let newDirectoryURL = self.documentsDirectory.appendingPathComponent(directoryName)
            do {
                try self.fileManager.createDirectory(at: newDirectoryURL, withIntermediateDirectories: false, attributes: nil)
                self.loadFiles()
                HapticFeedbackGenerator.generateNotificationFeedback(type: .success)
            } catch {
                self.utilities.handleError(in: self, error: error, withTitle: "Directory Creation Error")
            }
        }
        
        alertController.addAction(createAction)
        alertController.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        present(alertController, animated: true, completion: nil)
    }
    
    /// Imports a file using the file handler
    @objc func importFile() {
        fileHandlers.importFile(viewController: self)
    }
    
    /// Sorts the file list according to the current sort order
    func sortFiles() {
        switch sortOrder {
            case .name:
                fileList.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
            case .date:
                fileList.sort { $0.date > $1.date }
            case .size:
                fileList.sort { $0.size > $1.size }
        }
        
        // Also sort filtered file list if search is active
        if searchController.isActive {
            switch sortOrder {
                case .name:
                    filteredFileList.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
                case .date:
                    filteredFileList.sort { $0.date > $1.date }
                case .size:
                    filteredFileList.sort { $0.size > $1.size }
            }
        }
    }
    
    @objc func showMenu() {
        let alertController = UIAlertController(title: "Sort By", message: nil, preferredStyle: .actionSheet)
        
        let sortByNameAction = UIAlertAction(title: "Name", style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.sortOrder = .name
            self.sortFiles()
            HomeViewUI.fileListTableView.reloadData()
        }
        
        let sortByDateAction = UIAlertAction(title: "Date", style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.sortOrder = .date
            self.sortFiles()
            HomeViewUI.fileListTableView.reloadData()
        }
        
        let sortBySizeAction = UIAlertAction(title: "Size", style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.sortOrder = .size
            self.sortFiles()
            HomeViewUI.fileListTableView.reloadData()
        }
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        alertController.addAction(sortByNameAction)
        alertController.addAction(sortByDateAction)
        alertController.addAction(sortBySizeAction)
        alertController.addAction(cancelAction)

        // iPad support for action sheets
        if let popover = alertController.popoverPresentationController {
            if let barButtonItem = navigationItem.rightBarButtonItems?.first {
                popover.barButtonItem = barButtonItem
            } else {
                popover.sourceView = self.view
                popover.sourceRect = CGRect(x: self.view.bounds.midX, y: self.view.bounds.midY, width: 0, height: 0)
                popover.permittedArrowDirections = []
            }
        }

        present(alertController, animated: true, completion: nil)
    }
    
    func updateSearchResults(for searchController: UISearchController) {
        guard let searchText = searchController.searchBar.text?.lowercased() else { return }
        filteredFileList = fileList.filter { $0.name.lowercased().contains(searchText) }
        HomeViewUI.fileListTableView.reloadData()
    }
    
    /// Shows a message when the file list is empty
    /// - Parameter error: Optional error to show
    private func showEmptyStateMessage(withError error: Error? = nil) {
        // Check if we already have an empty state label
        if let existingLabel = view.viewWithTag(1001) as? UILabel {
            existingLabel.isHidden = false

            if let error = error {
                existingLabel.text = "Could not load files.\n\(error.localizedDescription)\n\nTap the upload button to add files."
            } else {
                existingLabel.text = "No files found.\n\nTap the upload button to add files."
            }
            return
        }

        // Create a new empty state label
        let emptyLabel = UILabel()
        emptyLabel.tag = 1001
        emptyLabel.translatesAutoresizingMaskIntoConstraints = false
        emptyLabel.textAlignment = .center
        emptyLabel.numberOfLines = 0
        emptyLabel.textColor = .secondaryLabel
        emptyLabel.font = .systemFont(ofSize: 16)

        if let error = error {
            emptyLabel.text = "Could not load files.\n\(error.localizedDescription)\n\nTap the upload button to add files."
        } else {
            emptyLabel.text = "No files found.\n\nTap the upload button to add files."
        }

        view.addSubview(emptyLabel)

        NSLayoutConstraint.activate([
            emptyLabel.centerXAnchor.constraint(equalTo: HomeViewUI.fileListTableView.centerXAnchor),
            emptyLabel.centerYAnchor.constraint(equalTo: HomeViewUI.fileListTableView.centerYAnchor),
            emptyLabel.leadingAnchor.constraint(equalTo: HomeViewUI.fileListTableView.leadingAnchor, constant: 40),
            emptyLabel.trailingAnchor.constraint(equalTo: HomeViewUI.fileListTableView.trailingAnchor, constant: -40),
        ])
    }

    /// Hides the empty state message
    private func hideEmptyStateMessage() {
        if let emptyLabel = view.viewWithTag(1001) {
            emptyLabel.isHidden = true
        }
    }

    // MARK: - UITableViewDragDelegate
    // Implementation moved to FileDragAndDrop.swift extension

    // MARK: - UITableViewDropDelegate
    // Implementation moved to FileDragAndDrop.swift extension

    // MARK: - FileHandlingDelegate

    override func present(_ viewControllerToPresent: UIViewController, animated flag: Bool, completion: (() -> Void)? = nil) {
        super.present(viewControllerToPresent, animated: flag, completion: completion)
    }
}