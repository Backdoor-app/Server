// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class SigningsAdvancedViewController: FRSITableViewCOntroller {
    private var toggleOptions: [TogglesOption]

    override init(signingDataWrapper: SigningDataWrapper, mainOptions: SigningMainDataWrapper) {
        self.toggleOptions = backdoor.toggleOptions(signingDataWrapper: signingDataWrapper)
        super.init(signingDataWrapper: signingDataWrapper, mainOptions: mainOptions)
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        tableData = [
            [String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_APPEARENCE")],
            [String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_MINIMUM_APP_VERSION")],
            [],
        ]

        sectionTitles = [
            String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_APPEARENCE"),
            String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_MINIMUM_APP_VERSION"),
            String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_PROPERTIES"),
        ]

        self.title = String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_PROPERTIES")
        self.tableData[2] = toggleOptions.map { $0.title }
    }
}

extension SigningsAdvancedViewController {
    override func tableView(_: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let reuseIdentifier = "Cell"
        let cell = UITableViewCell(style: .value1, reuseIdentifier: reuseIdentifier)
        cell.accessoryType = .none
        cell.selectionStyle = .gray

        let cellText = tableData[indexPath.section][indexPath.row]
        cell.textLabel?.text = cellText

        switch cellText {
            case String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_APPEARENCE"):
                let forceLightDarkAppearence = TweakLibraryViewCell()
                forceLightDarkAppearence.selectionStyle = .none
                forceLightDarkAppearence.configureSegmentedControl(
                    with: mainOptions.mainOptions.forceLightDarkAppearenceString,
                    selectedIndex: 0
                )
                forceLightDarkAppearence.segmentedControl.addTarget(self, action: #selector(forceLightDarkAppearenceDidChange(_:)), for: .valueChanged)

                return forceLightDarkAppearence
            case String.localized("APP_SIGNING_INPUT_VIEW_CONTROLLER_SECTION_TITLE_MINIMUM_APP_VERSION"):
                let forceMinimumVersion = TweakLibraryViewCell()
                forceMinimumVersion.selectionStyle = .none
                forceMinimumVersion.configureSegmentedControl(
                    with: mainOptions.mainOptions.forceMinimumVersionString,
                    selectedIndex: 0
                )
                forceMinimumVersion.segmentedControl.addTarget(self, action: #selector(forceMinimumVersionDidChange(_:)), for: .valueChanged)

                return forceMinimumVersion
            default:
                break
        }

        if indexPath.section == 2 {
            let toggleOption = toggleOptions[indexPath.row]
            cell.textLabel?.text = toggleOption.title
            let toggleSwitch = UISwitch()
            toggleSwitch.isOn = toggleOption.binding
            toggleSwitch.tag = indexPath.row
            toggleSwitch.addTarget(self, action: #selector(toggleOptionsSwitches(_:)), for: .valueChanged)
            cell.accessoryView = toggleSwitch
        }

        return cell
    }
}

extension SigningsAdvancedViewController {
    @objc private func forceLightDarkAppearenceDidChange(_ sender: UISegmentedControl) {
        signingDataWrapper.signingOptions.forceLightDarkAppearence =
            mainOptions.mainOptions.forceLightDarkAppearenceString[sender.selectedSegmentIndex]
    }

    @objc private func forceMinimumVersionDidChange(_ sender: UISegmentedControl) {
        signingDataWrapper.signingOptions.forceMinimumVersion =
            mainOptions.mainOptions.forceMinimumVersionString[sender.selectedSegmentIndex]
    }

    @objc func toggleOptionsSwitches(_ sender: UISwitch) {
        switch sender.tag {
            case 0:
                signingDataWrapper.signingOptions.removePlugins = sender.isOn
            case 1:
                signingDataWrapper.signingOptions.forceFileSharing = sender.isOn
            case 2:
                signingDataWrapper.signingOptions.removeSupportedDevices = sender.isOn
            case 3:
                signingDataWrapper.signingOptions.removeURLScheme = sender.isOn
            case 4:
                signingDataWrapper.signingOptions.forceProMotion = sender.isOn
            case 5:
                signingDataWrapper.signingOptions.forceForceFullScreen = sender.isOn
            case 6:
                signingDataWrapper.signingOptions.forceiTunesFileSharing = sender.isOn
            case 7:
                signingDataWrapper.signingOptions.forceTryToLocalize = sender.isOn
            case 8:
                signingDataWrapper.signingOptions.removeProvisioningFile = sender.isOn
            case 9:
                signingDataWrapper.signingOptions.removeWatchPlaceHolder = sender.isOn
            default:
                break
        }
    }
}
