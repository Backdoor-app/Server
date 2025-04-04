// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

enum DownloadState {
    case notStarted
    case inProgress(progress: CGFloat)
    case completed
    case failed(error: Error)

    var progress: CGFloat? {
        switch self {
            case let .inProgress(progress):
                return progress
            default:
                return nil
        }
    }
}

class DownloadTask {
    var uuid: String
    weak var cell: AppTableViewCell?
    var state: DownloadState
    var dl: AppDownload
    var progressHandler: ((CGFloat) -> Void)?

    init(uuid: String, cell: AppTableViewCell, state: DownloadState = .notStarted, dl: AppDownload) {
        self.uuid = uuid
        self.cell = cell
        self.state = state
        self.dl = dl
    }

    func updateProgress(to progress: CGFloat) {
        state = .inProgress(progress: progress)
        progressHandler?(progress)
        NotificationCenter.default.post(name: .downloadProgressUpdated, object: self, userInfo: ["uuid": uuid, "progress": progress])
    }
}

extension Notification.Name {
    static let downloadProgressUpdated = Notification.Name("downloadProgressUpdated")
}

class DownloadTaskManager {
    static let shared = DownloadTaskManager()
    public var downloadTasks: [String: DownloadTask] = [:]
    private let taskQueue = DispatchQueue(label: "com.backdoor.DownloadTaskManager", attributes: .concurrent)
    
    private init() {
        NotificationCenter.default.addObserver(self, selector: #selector(appWillTerminate), name: UIApplication.willTerminateNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(handleMemoryWarning), name: UIApplication.didReceiveMemoryWarningNotification, object: nil)
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    func addTask(uuid: String, cell: AppTableViewCell, dl: AppDownload) {
        let task = DownloadTask(uuid: uuid, cell: cell, dl: dl)
        taskQueue.async(flags: .barrier) { [weak self] in
            self?.downloadTasks[uuid] = task
        }
    }

    func updateTask(uuid: String, state: DownloadState) {
        taskQueue.async { [weak self] in
            guard let self = self, let task = self.downloadTasks[uuid] else { return }
            
            task.state = state
            self.persistTaskState(task)
            
            DispatchQueue.main.async {
                switch state {
                    case let .inProgress(progress):
                        task.cell?.updateProgress(to: progress)
                    case .completed, .failed:
                        task.cell?.stopDownload()
                        self.removeTask(uuid: uuid)
                        self.removePersistedTaskState(for: uuid)
                    default:
                        break
                }
            }
        }
    }

    func cancelDownload(for uuid: String) {
        taskQueue.async { [weak self] in
            guard let self = self, let task = self.downloadTasks[uuid] else { return }
            
            task.dl.cancelDownload()
            
            DispatchQueue.main.async {
                task.cell?.cancelDownload()
            }
            
            self.removeTask(uuid: uuid)
        }
    }

    func updateTaskProgress(uuid: String, progress: CGFloat) {
        taskQueue.async { [weak self] in
            guard let task = self?.downloadTasks[uuid] else { return }
            task.updateProgress(to: progress)
        }
    }

    func removeTask(uuid: String) {
        taskQueue.async(flags: .barrier) { [weak self] in
            self?.downloadTasks.removeValue(forKey: uuid)
            self?.removePersistedTaskState(for: uuid)
        }
    }

    func task(for uuid: String) -> DownloadTask? {
        var result: DownloadTask?
        taskQueue.sync {
            result = downloadTasks[uuid]
        }
        return result
    }

    private func persistTaskState(_ task: DownloadTask) {
        let defaults = UserDefaults.standard
        defaults.set(task.state.progress, forKey: "\(task.uuid)_progress")
    }

    private func removePersistedTaskState(for uuid: String) {
        let defaults = UserDefaults.standard
        defaults.removeObject(forKey: "\(uuid)_progress")
    }

    func restoreTaskState(for uuid: String, cell: AppTableViewCell) {
        taskQueue.async { [weak self] in
            guard let self = self, let task = self.downloadTasks[uuid] else { return }
            
            let defaults = UserDefaults.standard
            if let progress = defaults.value(forKey: "\(uuid)_progress") as? CGFloat {
                let updatedTask = DownloadTask(uuid: uuid, cell: cell, state: .inProgress(progress: progress), dl: task.dl)
                self.downloadTasks[uuid] = updatedTask
            }
        }
    }

    @objc private func appWillTerminate() {
        clearAllTasks()
    }
    
    @objc private func handleMemoryWarning() {
        // Clean up any completed or failed tasks that might still be in memory
        taskQueue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            // Filter out tasks with nil cells (recycled cells)
            let tasksToRemove = self.downloadTasks.filter { $0.value.cell == nil }
            for (uuid, _) in tasksToRemove {
                self.downloadTasks.removeValue(forKey: uuid)
                self.removePersistedTaskState(for: uuid)
            }
        }
    }

    private func clearAllTasks() {
        taskQueue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            let defaults = UserDefaults.standard
            for uuid in self.downloadTasks.keys {
                defaults.removeObject(forKey: "\(uuid)_progress")
            }
            defaults.synchronize()
            self.downloadTasks.removeAll()
        }
    }
}
