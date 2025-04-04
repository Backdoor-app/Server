// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import ZIPFoundation

class DirectoryViewController: HomeViewController {
    // MARK: - Properties

    /// The directory URL this controller is showing
    private var directoryURL: URL

    /// Callback to notify parent when changes occur
    var onContentChanged: (() -> Void)?

    // MARK: - Initialization

    /// Initialize with a directory URL
    /// - Parameter directory: The URL of the directory to display
    init(directory: URL) {
        self.directoryURL = directory
        super.init(nibName: nil, bundle: nil)
        self.title = directory.lastPathComponent
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupFileManagementUI() // Call the setup method from the extension
    }

    // MARK: - Overrides

    /// Override documentsDirectory to use the specified directory URL
    override var documentsDirectory: URL {
        return directoryURL
    }

    /// Load files from the directory URL
    override func loadFiles() {
        super.loadFiles()
    }

    /// Reload content when returning to view
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        loadFiles()
    }
}
