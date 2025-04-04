// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class DisplayViewController: FRSTableViewController {
    let collectionData = ["Lime Green", "Red", "Purple", "Baby Blue", "Yellow", "White", "Orange"]
    let collectionDataColors = ["99CC00", "FF0000", "800080", "89CFF0", "FFFF00", "FFFFFF", "FFA500"]

    override func viewDidLoad() {
        super.viewDidLoad()
        title = String.localized("SETTINGS_VIEW_CONTROLLER_CELL_DISPLAY")

        tableData = [
            [String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_APPEARENCE")],
            ["Collection View"],
            [],
            ["Certificate Name"],
        ]

        sectionTitles = [
            "",
            String.localized("DISPLAY_VIEW_CONTROLLER_SECTION_TITLE_TINT_COLOR"),
            String.localized("DISPLAY_VIEW_CONTROLLER_SECTION_TITLE_STORE"),
            String.localized("CERTIFICATES_VIEW_CONTROLLER_TITLE"),
        ]

        self.tableView.register(CollectionTableViewCell.self, forCellReuseIdentifier: "CollectionCell")
    }

    private func updateAppearance(with style: UIUserInterfaceStyle) {
        view.window?.overrideUserInterfaceStyle = style
        Preferences.preferredInterfaceStyle = style.rawValue
    }
}

extension DisplayViewController {
    override func tableView(_: UITableView, numberOfRowsInSection section: Int) -> Int {
        if section == 2 {
            return 3
        } else {
            return tableData[section].count
        }
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let reuseIdentifier = "Cell"
        var cell = UITableViewCell(style: .default, reuseIdentifier: reuseIdentifier)
        cell.selectionStyle = .none

        if indexPath.section == 2 {
            cell = UITableViewCell(style: .subtitle, reuseIdentifier: reuseIdentifier)
            switch indexPath.row {
                case 0:
                    cell.textLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_DEFAULT_SUBTITLE")
                    cell.detailTextLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_DEFAULT_SUBTITLE_DESCRIPTION")
                case 1:
                    cell.textLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_LOCALIZED_SUBTITLE")
                    cell.detailTextLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_LOCALIZED_SUBTITLE_DESCRIPTION")
                case 2:
                    cell.textLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_BIG_DESCRIPTION")
                    cell.detailTextLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_BIG_DESCRIPTION_DESCRIPTION")
                default:
                    break
            }
            cell.textLabel?.font = UIFont.boldSystemFont(ofSize: UIFont.systemFontSize)
            cell.detailTextLabel?.textColor = .secondaryLabel
            cell.textLabel?.numberOfLines = 0
            cell.detailTextLabel?.numberOfLines = 0

            if Preferences.appDescriptionAppearence == indexPath.row {
                cell.accessoryType = .checkmark
            } else {
                cell.accessoryType = .none
            }
            return cell
        }

        let cellText = tableData[indexPath.section][indexPath.row]
        switch cellText {
            case String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_APPEARENCE"):
                cell.textLabel?.text = String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_APPEARENCE")
                let segmentedControl = UISegmentedControl(items: UIUserInterfaceStyle.allCases.map { $0.description })
                segmentedControl.selectedSegmentIndex = UIUserInterfaceStyle.allCases.firstIndex { $0.rawValue == Preferences.preferredInterfaceStyle } ?? 0
                segmentedControl.addTarget(self, action: #selector(appearanceSegmentedControlChanged(_:)), for: .valueChanged)
                cell.accessoryView = segmentedControl
            case "Collection View":
                let cell = tableView.dequeueReusableCell(withIdentifier: "CollectionCell", for: indexPath) as! CollectionTableViewCell
                cell.setData(collectionData: collectionData, colors: collectionDataColors)
                cell.backgroundColor = .clear
                return cell
            case "Certificate Name":
                let useTeamName = SwitchViewCell()
                useTeamName.textLabel?.text = String.localized("DISPLAY_VIEW_CONTROLLER_CELL_TEAM_NAME")
                useTeamName.switchControl.addTarget(self, action: #selector(certificateNameToggle(_:)), for: .valueChanged)
                useTeamName.switchControl.isOn = Preferences.certificateTitleAppIDtoTeamID
                useTeamName.selectionStyle = .none
                return useTeamName
            default:
                break
        }

        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        if indexPath.section == 2 {
            let previousSelection = Preferences.appDescriptionAppearence
            Preferences.appDescriptionAppearence = indexPath.row

            let previousIndexPath = IndexPath(row: previousSelection, section: indexPath.section)
            tableView.reloadRows(at: [previousIndexPath, indexPath], with: .fade)
        }
    }

    override func tableView(_: UITableView, titleForFooterInSection section: Int) -> String? {
        switch section {
            case 3:
                return String.localized("DISPLAY_VIEW_CONTROLLER_CELL_TEAM_NAME_DESCRIPTION")
            default:
                return nil
        }
    }

    @objc private func appearanceSegmentedControlChanged(_ sender: UISegmentedControl) {
        let selectedStyle = UIUserInterfaceStyle.allCases[sender.selectedSegmentIndex]
        updateAppearance(with: selectedStyle)
    }

    @objc private func certificateNameToggle(_ sender: UISwitch) {
        Preferences.certificateTitleAppIDtoTeamID = sender.isOn
    }
}
