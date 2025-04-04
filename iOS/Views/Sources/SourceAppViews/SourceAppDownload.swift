// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import AlertKit
import Foundation
import Nuke
import UIKit

extension SourceAppViewController: DownloadDelegate {
    func stopDownload(uuid: String) {
        DispatchQueue.main.async {
            if let task = DownloadTaskManager.shared.task(for: uuid) {
                if let cell = task.cell {
                    cell.stopDownload()
                }
                DownloadTaskManager.shared.removeTask(uuid: uuid)
            }
        }
    }

    func startDownload(uuid: String, indexPath _: IndexPath) {
        DispatchQueue.main.async {
            if let task = DownloadTaskManager.shared.task(for: uuid) {
                if let cell = task.cell {
                    cell.startDownload()
                }
                DownloadTaskManager.shared.updateTask(uuid: uuid, state: .inProgress(progress: 0.0))
            }
        }
    }

    func updateDownloadProgress(progress: Double, uuid: String) {
        DownloadTaskManager.shared.updateTask(uuid: uuid, state: .inProgress(progress: progress))
    }
}

extension SourceAppViewController {
    func startDownloadIfNeeded(for indexPath: IndexPath, in tableView: UITableView, downloadURL: URL?, appUUID: String?, sourceLocation: String) {
        guard let downloadURL = downloadURL, let appUUID = appUUID, let cell = tableView.cellForRow(at: indexPath) as? AppTableViewCell else {
            return
        }

        if cell.appDownload == nil {
            cell.appDownload = AppDownload()
            cell.appDownload?.dldelegate = self
        }
        
        // Show download animation in cell
        let animationView = cell.addAnimatedIcon(
            systemName: "arrow.down.circle",
            tintColor: .systemBlue,
            size: CGSize(width: 40, height: 40)
        )
        
        // Position animation in the cell
        animationView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            animationView.trailingAnchor.constraint(equalTo: cell.contentView.trailingAnchor, constant: -16),
            animationView.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
            animationView.widthAnchor.constraint(equalToConstant: 40),
            animationView.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Add to task manager
        DownloadTaskManager.shared.addTask(uuid: appUUID, cell: cell, dl: cell.appDownload!)

        // Use NetworkManager to handle the download with improved error handling
        Task {
            do {
                // Create a temporary file path for the download
                let tempDir = FileManager.default.temporaryDirectory
                let filePath = tempDir.appendingPathComponent("app_\(appUUID).ipa")
                
                // Start download and show progress
                self.startDownload(uuid: appUUID, indexPath: indexPath)
                
                // Download file with URLSession
                var request = URLRequest(url: downloadURL)
                let (tempFileURL, _) = try await URLSession.shared.download(for: request)
                try FileManager.default.moveItem(at: tempFileURL, to: filePath)
                let downloadedURL = filePath
                
                // Verify downloaded file integrity
                let fileData = try Data(contentsOf: downloadedURL)
                let checksum = CryptoHelper.shared.crc32(of: fileData)
                Debug.shared.log(message: "Download completed with checksum: \(checksum)", type: .info)
                
                // Extract and process the bundle - removed unused self capture
                cell.appDownload?.extractCompressedBundle(packageURL: downloadedURL.path) { targetBundle, error in
                    
                    // Remove animation when processing is complete
                    DispatchQueue.main.async {
                        animationView.removeFromSuperview()
                    }
                    
                    if let error = error {
                        DownloadTaskManager.shared.updateTask(uuid: appUUID, state: .failed(error: error))
                        Debug.shared.log(message: "Extraction error: \(error.localizedDescription)", type: .error)
                        
                        // Show error animation
                        let errorAnimation = cell.addAnimatedIcon(
                            systemName: "exclamationmark.circle",
                            tintColor: .systemRed,
                            size: CGSize(width: 40, height: 40)
                        )
                        
                        errorAnimation.translatesAutoresizingMaskIntoConstraints = false
                        NSLayoutConstraint.activate([
                            errorAnimation.trailingAnchor.constraint(equalTo: cell.contentView.trailingAnchor, constant: -16),
                            errorAnimation.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
                            errorAnimation.widthAnchor.constraint(equalToConstant: 40),
                            errorAnimation.heightAnchor.constraint(equalToConstant: 40)
                        ])
                        
                        // Remove error animation after delay
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            errorAnimation.removeFromSuperview()
                        }
                    } else if let targetBundle = targetBundle {
                        cell.appDownload?.addToApps(bundlePath: targetBundle, uuid: appUUID, sourceLocation: sourceLocation) { error in
                            if let error = error {
                                DownloadTaskManager.shared.updateTask(uuid: appUUID, state: .failed(error: error))
                                Debug.shared.log(message: "Failed to add app: \(error.localizedDescription)", type: .error)
                            } else {
                                DownloadTaskManager.shared.updateTask(uuid: appUUID, state: .completed)
                                Debug.shared.log(message: "Done", type: .success)
                                
                                // Show success animation
                                let successAnimation = cell.addAnimatedIcon(
                                    systemName: "checkmark.circle",
                                    tintColor: .systemGreen,
                                    size: CGSize(width: 40, height: 40)
                                )
                                
                                successAnimation.translatesAutoresizingMaskIntoConstraints = false
                                NSLayoutConstraint.activate([
                                    successAnimation.trailingAnchor.constraint(equalTo: cell.contentView.trailingAnchor, constant: -16),
                                    successAnimation.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
                                    successAnimation.widthAnchor.constraint(equalToConstant: 40),
                                    successAnimation.heightAnchor.constraint(equalToConstant: 40)
                                ])
                                
                                // Remove success animation after delay
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    successAnimation.removeFromSuperview()
                                }

                                // Check if immediate install is enabled
                                if UserDefaults.standard.signingOptions.immediatelyInstallFromSource {
                                    DispatchQueue.main.async {
                                        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
                                        if let downloadedApp = downloadedApps.first(where: { $0.uuid == appUUID }) {
                                            NotificationCenter.default.post(
                                                name: Notification.Name("InstallDownloadedApp"),
                                                object: nil,
                                                userInfo: ["downloadedApp": downloadedApp]
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } catch {
                // Handle download errors with enhanced error reporting
                DownloadTaskManager.shared.updateTask(uuid: appUUID, state: .failed(error: error))
                
                // Remove animation
                DispatchQueue.main.async {
                    animationView.removeFromSuperview()
                }
                
                // Log detailed error information
                if let networkError = error as? NetworkError {
                    Debug.shared.log(message: "Network download error: \(networkError.localizedDescription)", type: .error)
                    
                    // Add detailed error diagnostics
                    switch networkError {
                    case .httpError(let statusCode):
                        Debug.shared.log(message: "HTTP error status: \(statusCode)", type: .error)
                    case .invalidURL:
                        Debug.shared.log(message: "Invalid download URL: \(downloadURL)", type: .error)
                    default:
                        Debug.shared.log(message: "Download failed with error: \(error.localizedDescription)", type: .error)
                    }
                } else {
                    Debug.shared.log(message: "Download failed: \(error.localizedDescription)", type: .error)
                }
                
                // Show error animation
                let errorAnimation = cell.addAnimatedIcon(
                    systemName: "exclamationmark.circle",
                    tintColor: .systemRed,
                    size: CGSize(width: 40, height: 40)
                )
                
                errorAnimation.translatesAutoresizingMaskIntoConstraints = false
                NSLayoutConstraint.activate([
                    errorAnimation.trailingAnchor.constraint(equalTo: cell.contentView.trailingAnchor, constant: -16),
                    errorAnimation.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
                    errorAnimation.widthAnchor.constraint(equalToConstant: 40),
                    errorAnimation.heightAnchor.constraint(equalToConstant: 40)
                ])
                
                // Remove error animation after delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    errorAnimation.removeFromSuperview()
                }
            }
        }
    }
}

protocol DownloadDelegate: AnyObject {
    func updateDownloadProgress(progress: Double, uuid: String)
    func stopDownload(uuid: String)
}

// This extension is moved to UIApplication+TopViewController.swift to avoid redeclaration
