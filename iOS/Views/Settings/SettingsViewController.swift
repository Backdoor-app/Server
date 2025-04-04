// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Nuke
import SwiftUI
import UIKit

class SettingsViewController: FRSTableViewController {
    let aboutSection = [
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ABOUT", arguments: "Backdoor"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SUBMIT_FEEDBACK"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_GITHUB"),
    ]

    let displaySection = [
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_DISPLAY"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APP_ICON"),
    ]

    let certificateSection = [
        "Current Certificate",
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ADD_CERTIFICATES"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SIGN_OPTIONS"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SERVER_OPTIONS"),
    ]
    
    let aiSection = [
        "AI Learning Settings",
    ]

    let logsSection = [
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_VIEW_LOGS"),
    ]

    let foldersSection = [
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APPS_FOLDER"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_CERTS_FOLDER"),
    ]

    let resetSection = [
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_RESET"),
        String.localized("SETTINGS_VIEW_CONTROLLER_CELL_RESET_ALL"),
    ]

    // Flag to prevent double initialization
    private var isInitialized = false

    override func viewDidLoad() {
        super.viewDidLoad()

        // Defensive programming - ensure we're on the main thread for UI setup
        if !Thread.isMainThread {
            Debug.shared.log(message: "SettingsViewController.viewDidLoad called off main thread, dispatching to main", type: .error)
            DispatchQueue.main.async { [weak self] in
                self?.viewDidLoad()
            }
            return
        }

        // Set the title immediately for better user experience
        self.title = String.localized("TAB_SETTINGS")

        // Safety check against crashes during initialization
        do {
            initializeTableData()
            setupNavigation()

            // Mark as initialized
            isInitialized = true

            Debug.shared.log(message: "SettingsViewController initialized successfully", type: .info)
        } catch {
            Debug.shared.log(message: "Error initializing SettingsViewController: \(error.localizedDescription)", type: .error)

            // Show an error dialog if initialization fails
            let alert = UIAlertController(
                title: "Settings Error",
                message: "There was a problem loading settings. Please try again.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true, completion: nil)
        }
    }

    // Separate method for initialization to make error handling clearer
    private func initializeTableData() {
        tableData = [
            aboutSection,
            displaySection,
            certificateSection,
            aiSection,
            logsSection,
            foldersSection,
            resetSection,
        ]

        sectionTitles = ["", "", "", "", "", "", ""]
        ensureTableDataHasSections()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        // Only reload if already initialized to prevent crashes
        if isInitialized {
            self.tableView.reloadData()
        } else {
            // If not initialized yet, trigger viewDidLoad again
            viewDidLoad()
        }
    }

    fileprivate func setupNavigation() {
        self.title = String.localized("TAB_SETTINGS")

        // Ensure the navigation bar is properly configured
        if let navController = navigationController {
            navController.navigationBar.prefersLargeTitles = true
            navController.navigationBar.tintColor = Preferences.appTintColor.uiColor
        }
    }

    // MARK: - ViewControllerRefreshable

    override func refreshContent() {
        // Only refresh if view is loaded and initialized
        if isViewLoaded && isInitialized {
            tableView.reloadData()
        }
    }

    // MARK: - UITableViewDataSource & UITableViewDelegate overrides

    override func tableView(_: UITableView, numberOfRowsInSection section: Int) -> Int {
        // Safety check to prevent crashes
        guard isInitialized, section < tableData.count else {
            return 0
        }
        return tableData[section].count
    }

    override func numberOfSections(in _: UITableView) -> Int {
        // Safety check to prevent crashes
        guard isInitialized else {
            return 0
        }
        return tableData.count
    }
}

extension SettingsViewController {
    override func tableView(_: UITableView, titleForFooterInSection section: Int) -> String? {
        if Preferences.beta, section == 0 {
            return String.localized("SETTINGS_VIEW_CONTROLLER_SECTION_FOOTER_ISSUES")
        } else if !Preferences.beta, section == 1 {
            return String.localized("SETTINGS_VIEW_CONTROLLER_SECTION_FOOTER_ISSUES")
        }

        switch section {
            case sectionTitles.count - 1: 
                let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
                let buildNumber = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown"
                return "Backdoor \(appVersion) (\(buildNumber)) • iOS \(UIDevice.current.systemVersion)"
            default:
                return nil
        }
    }

    override func tableView(_: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let reuseIdentifier = "Cell"
        let cell = UITableViewCell(style: .value1, reuseIdentifier: reuseIdentifier)
        cell.accessoryType = .none
        cell.selectionStyle = .none

        let cellText = tableData[indexPath.section][indexPath.row]
        cell.textLabel?.text = cellText

        switch cellText {
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ABOUT", arguments: "Backdoor"):
                cell.setAccessoryIcon(with: "info.circle")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SUBMIT_FEEDBACK"), String.localized("SETTINGS_VIEW_CONTROLLER_CELL_GITHUB"):
                cell.textLabel?.textColor = .tintColor
                cell.setAccessoryIcon(with: "safari")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_DISPLAY"):
                cell.setAccessoryIcon(with: "paintbrush")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APP_ICON"):
                cell.setAccessoryIcon(with: "app.dashed")
                cell.selectionStyle = .default

            case "Current Certificate":
                if let hasGotCert = CoreDataManager.shared.getCurrentCertificate() {
                    let cell = CertificateViewTableViewCell()
                    cell.configure(with: hasGotCert, isSelected: false)
                    cell.selectionStyle = .none
                    return cell
                } else {
                    cell.textLabel?.text = String.localized("SETTINGS_VIEW_CONTROLLER_CELL_CURRENT_CERTIFICATE_NOSELECTED")
                    cell.textLabel?.textColor = .secondaryLabel
                    cell.selectionStyle = .none
                }

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ADD_CERTIFICATES"):
                cell.setAccessoryIcon(with: "plus")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SIGN_OPTIONS"):
                cell.setAccessoryIcon(with: "signature")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SERVER_OPTIONS"):
                cell.setAccessoryIcon(with: "server.rack")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_VIEW_LOGS"):
                cell.setAccessoryIcon(with: "newspaper")
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APPS_FOLDER"),
                 String.localized("SETTINGS_VIEW_CONTROLLER_CELL_CERTS_FOLDER"):
                cell.accessoryType = .disclosureIndicator
                cell.textLabel?.textColor = .tintColor
                cell.selectionStyle = .default

            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_RESET"),
                 String.localized("SETTINGS_VIEW_CONTROLLER_CELL_RESET_ALL"):
                cell.textLabel?.textColor = .tintColor
                cell.accessoryType = .disclosureIndicator
                cell.selectionStyle = .default

            default:
                break
        }

        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let itemTapped = tableData[indexPath.section][indexPath.row]
        switch itemTapped {
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ABOUT", arguments: "Backdoor"):
                let l = AboutViewController()
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_GITHUB"):
                guard let url = URL(string: "https://github.com/khcrysalis/Backdoor") else {
                    Debug.shared.log(message: "Invalid URL")
                    return
                }
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SUBMIT_FEEDBACK"):
                guard let url = URL(string: "https://github.com/khcrysalis/Backdoor/issues") else {
                    Debug.shared.log(message: "Invalid URL")
                    return
                }
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_DISPLAY"):
                let l = DisplayViewController()
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APP_ICON"):
                let l = IconsListViewController()
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ADD_CERTIFICATES"):
                let l = CertificatesViewController()
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SIGN_OPTIONS"):
                let signingDataWrapper = SigningDataWrapper(signingOptions: UserDefaults.standard.signingOptions)
                let l = SigningsOptionViewController(signingDataWrapper: signingDataWrapper)
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_SERVER_OPTIONS"):
                let l = ServerOptionsViewController()
                navigationController?.pushViewController(l, animated: true)
            case "AI Learning Settings":
                let l = AILearningSettingsViewController(style: .grouped)
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_VIEW_LOGS"):
                let l = LogsViewController()
                navigationController?.pushViewController(l, animated: true)
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APPS_FOLDER"):
                openDirectory(named: "Apps")
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_CERTS_FOLDER"):
                openDirectory(named: "Certificates")
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_RESET"):
                self.resetOptionsAction()
            case String.localized("SETTINGS_VIEW_CONTROLLER_CELL_RESET_ALL"):
                self.resetAllAction()
            default:
                break
        }

        tableView.deselectRow(at: indexPath, animated: true)
    }
}

extension UITableViewCell {
    func setAccessoryIcon(with symbolName: String, tintColor: UIColor = .tertiaryLabel, renderingMode: UIImage.RenderingMode = .alwaysOriginal) {
        if let image = UIImage(systemName: symbolName)?.withTintColor(tintColor, renderingMode: renderingMode) {
            let imageView = UIImageView(image: image)
            self.accessoryView = imageView
        } else {
            self.accessoryView = nil
        }
    }
}

private extension SettingsViewController {
    func openDirectory(named directoryName: String) {
        let directoryURL = getDocumentsDirectory().appendingPathComponent(directoryName)
        let path = directoryURL.absoluteString.replacingOccurrences(of: "file://", with: "shareddocuments://")

        UIApplication.shared.open(URL(string: path)!, options: [:]) { success in
            if success {
                Debug.shared.log(message: "File opened successfully.")
            } else {
                Debug.shared.log(message: "Failed to open file.")
            }
        }
    }
}
