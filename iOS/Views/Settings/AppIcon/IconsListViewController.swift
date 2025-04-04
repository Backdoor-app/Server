// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class IconsListViewController: UITableViewController {
    public class func altImage(_ name: String) -> UIImage {
        let path = Bundle.main.bundleURL.appendingPathComponent(name + "@2x.png")
        return UIImage(contentsOfFile: path.path) ?? UIImage()
    }

    var sections: [String: [AltIcon]] = [
        "Main": [
            AltIcon(displayName: "Backdoor", author: "Samara", key: nil, image: altImage("AppIcon60x60")),
            AltIcon(displayName: "macOS Backdoor", author: "Samara", key: "Mac", image: altImage("Mac")),
            AltIcon(displayName: "Evil Backdoor", author: "Samara", key: "Evil", image: altImage("Evil")),
            AltIcon(displayName: "Classic Backdoor", author: "Samara", key: "Early", image: altImage("Early")),
        ],
        "Wingio": [
            AltIcon(displayName: "Backdoor", author: "Wingio", key: "Wing", image: altImage("Wing")),
        ],
    ]

    init() { super.init(style: .insetGrouped) }
    @available(*, unavailable)
    required init?(coder _: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupViews()
        setupNavigation()
    }

    fileprivate func setupViews() {
        self.tableView.delegate = self
        self.tableView.dataSource = self
        self.tableView.rowHeight = 75
    }

    fileprivate func setupNavigation() {
        self.title = String.localized("SETTINGS_VIEW_CONTROLLER_CELL_APP_ICON")
        self.navigationItem.largeTitleDisplayMode = .never
    }

    private func sectionTitles() -> [String] {
        return Array(sections.keys).sorted()
    }

    private func icons(forSection section: Int) -> [AltIcon] {
        let title = sectionTitles()[section]
        return sections[title] ?? []
    }
}

extension IconsListViewController {
    override func numberOfSections(in _: UITableView) -> Int { return sectionTitles().count }
    override func tableView(_: UITableView, numberOfRowsInSection section: Int) -> Int { return icons(forSection: section).count }
    override func tableView(_: UITableView, heightForHeaderInSection _: Int) -> CGFloat { return 40 }

    override func tableView(_: UITableView, viewForHeaderInSection section: Int) -> UIView? {
        let title = sectionTitles()[section]
        let headerView = InsetGroupedSectionHeader(title: title)
        return headerView
    }

    override func tableView(_: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = IconsListTableViewCell()
        let icon = icons(forSection: indexPath.section)[indexPath.row]
        cell.altIcon = icon
        if UIApplication.shared.alternateIconName == icon.key {
            cell.accessoryType = .checkmark
        } else {
            cell.accessoryType = .none
        }
        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        let icon = icons(forSection: indexPath.section)[indexPath.row]

        // Store current selection for UI updates even if async call hasn't completed
        let selectedIconKey = icon.key

        // Show activity indicator
        let activityIndicator = UIActivityIndicatorView(style: .medium)
        activityIndicator.startAnimating()
        let cell = tableView.cellForRow(at: indexPath)
        cell?.accessoryView = activityIndicator

        // Set icon with proper error handling
        UIApplication.shared.setAlternateIconName(selectedIconKey) { [weak self] error in
            guard let self = self else { return }

            // Remove activity indicator
            cell?.accessoryView = nil

            if let error = error {
                // Log error but don't show to user (silent operation)
                Debug.shared.log(message: "Icon change error: \(error.localizedDescription)", type: .error)

                // Revert UI to previous state
                self.tableView.reloadData()
            } else {
                // Success - update all visible rows to reflect new selection
                if let visibleRows = self.tableView.indexPathsForVisibleRows {
                    for visiblePath in visibleRows {
                        let visibleCell = self.tableView.cellForRow(at: visiblePath) as? IconsListTableViewCell
                        let rowIcon = self.icons(forSection: visiblePath.section)[visiblePath.row]

                        if rowIcon.key == selectedIconKey {
                            visibleCell?.accessoryType = .checkmark
                        } else {
                            visibleCell?.accessoryType = .none
                        }
                    }
                }

                // Provide haptic feedback for confirmation
                let generator = UIImpactFeedbackGenerator(style: .medium)
                generator.prepare()
                generator.impactOccurred()
            }
        }
    }
}

struct AltIcon {
    var displayName: String
    var author: String
    var key: String?
    var image: UIImage
}
