// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import QuickLook
import UIKit

class FilePreviewController: UIViewController, QLPreviewControllerDataSource, QLPreviewControllerDelegate {
    // MARK: - Properties

    /// The file URL to preview
    private let fileURL: URL

    /// The preview controller
    private let previewController = QLPreviewController()

    // MARK: - Initialization

    /// Initialize with a file URL
    /// - Parameter fileURL: The URL of the file to preview
    init(fileURL: URL) {
        self.fileURL = fileURL
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
        setupPreviewController()
    }

    // MARK: - Setup

    /// Sets up the preview controller
    private func setupPreviewController() {
        previewController.dataSource = self
        previewController.delegate = self
        addChild(previewController)
        view.addSubview(previewController.view)
        previewController.view.frame = view.bounds
        previewController.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        previewController.didMove(toParent: self)
    }

    // MARK: - QLPreviewControllerDataSource

    func numberOfPreviewItems(in _: QLPreviewController) -> Int {
        return 1
    }

    func previewController(_: QLPreviewController, previewItemAt _: Int) -> QLPreviewItem {
        return fileURL as QLPreviewItem
    }

    // MARK: - QLPreviewControllerDelegate

    func previewController(_: QLPreviewController, didUpdateContentsOf _: QLPreviewItem) {
        // Handle any updates if needed
    }
}
// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

//

//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

//
// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
//

//
// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
//
