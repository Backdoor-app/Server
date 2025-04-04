// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

enum FileAppError: Error {
    case fileNotFound(String)
    case fileAlreadyExists(String)
    case invalidFileName(String)
    case invalidFileType(String)
    case permissionDenied(String)
    case directoryCreationFailed(String)
    case fileCreationFailed(String)
    case fileRenameFailed(String, String)
    case fileDeleteFailed(String)
    case fileMoveFailed(String, String)
    case fileUnzipFailed(String, String, Error?)
    case fileZipFailed(String, String, Error?)
    case dylibListingFailed(String, Error?)
    case unknown(Error)
}

struct AlertConfig {
    let title: String?
    let message: String?
    let style: UIAlertController.Style
    let actions: [AlertActionConfig]
    let preferredAction: Int?
    let completionHandler: (() -> Void)?
}

struct AlertActionConfig {
    let title: String?
    let style: UIAlertAction.Style
    let handler: (() -> Void)?
}

class HomeViewUtilities {
    func handleError(in viewController: UIViewController, error: Error, withTitle title: String) {
        var message: String
        switch error {
            case let fileError as FileAppError:
                message = formatFileAppError(fileError)
            case let nsError as NSError:
                message = nsError.localizedDescription
                Debug.shared.log(message: "NSError: \(nsError.localizedDescription)", type: .error)
            default:
                message = error.localizedDescription
                Debug.shared.log(message: "Unknown error: \(error.localizedDescription)", type: .error)
        }

        DispatchQueue.main.async {
            let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
            viewController.present(alert, animated: true, completion: nil)
        }
    }

    private func formatFileAppError(_ error: FileAppError) -> String {
        switch error {
            case let .fileNotFound(fileName):
                Debug.shared.log(message: "File not found: \(fileName)", type: .info)
                return "File not found: \(fileName). Please check the file name."
            case let .fileAlreadyExists(fileName):
                Debug.shared.log(message: "File already exists: \(fileName)", type: .info)
                return "A file named \(fileName) already exists. Choose a different name."
            case let .unknown(underlyingError):
                Debug.shared.log(message: "Unknown error: \(underlyingError.localizedDescription)", type: .error)
                return "An unknown error occurred: \(underlyingError.localizedDescription)"
            default:
                return error.localizedDescription
        }
    }
}

extension UIViewController {
    func presentAlert(config: AlertConfig) {
        let alert = UIAlertController(title: config.title, message: config.message, preferredStyle: config.style)

        if let preferredActionIndex = config.preferredAction, preferredActionIndex < config.actions.count {
            alert.preferredAction = alert.actions[preferredActionIndex]
        }

        for actionConfig in config.actions {
            let action = UIAlertAction(title: actionConfig.title, style: actionConfig.style) { _ in
                actionConfig.handler?()
            }
            alert.addAction(action)
        }

        DispatchQueue.main.async {
            self.present(alert, animated: true, completion: config.completionHandler)
        }
    }
}

class HapticFeedbackGenerator {
    static func generateHapticFeedback(style: UIImpactFeedbackGenerator.FeedbackStyle) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.prepare()
        generator.impactOccurred()
    }

    static func generateNotificationFeedback(type: UINotificationFeedbackGenerator.FeedbackType) {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(type)
    }
}
