// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

class SigningsInputViewController: UITableViewController {
    var parentView: SigningsViewController
    var initialValue: String
    var valueToSaveTo: Int
    private var changedValue: String?

    private lazy var textField: UITextField = {
        let textField = UITextField(frame: .zero)
        textField.translatesAutoresizingMaskIntoConstraints = false
        textField.addTarget(self, action: #selector(textDidChange), for: .editingChanged)
        return textField
    }()

    init(parentView: SigningsViewController, initialValue: String, valueToSaveTo: Int) {
        self.parentView = parentView
        self.initialValue = initialValue
        self.valueToSaveTo = valueToSaveTo
        super.init(style: .insetGrouped)
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        navigationItem.largeTitleDisplayMode = .never
        self.title = initialValue.capitalized

        let saveButton = UIBarButtonItem(title: String.localized("SAVE"), style: .done, target: self, action: #selector(saveButton))
        saveButton.isEnabled = false
        navigationItem.rightBarButtonItem = saveButton

        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "InputCell")
    }

    @objc func saveButton() {
        switch valueToSaveTo {
            case 1:
                parentView.mainOptions.mainOptions.name = changedValue
            case 2:
                parentView.mainOptions.mainOptions.bundleId = changedValue
            case 3:
                parentView.mainOptions.mainOptions.version = changedValue
            default:
                break
        }

        self.navigationController?.popViewController(animated: true)
    }

    @objc private func textDidChange() {
        navigationItem.rightBarButtonItem?.isEnabled = !(textField.text?.isEmpty ?? true)
        changedValue = textField.text
    }

    override func numberOfSections(in _: UITableView) -> Int { return 1 }
    override func tableView(_: UITableView, numberOfRowsInSection _: Int) -> Int { return 1 }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "InputCell", for: indexPath)
        switch indexPath.section {
            case 0:
                textField.text = initialValue
                textField.placeholder = initialValue

                if textField.superview == nil {
                    cell.contentView.addSubview(textField)
                    NSLayoutConstraint.activate([
                        textField.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
                        textField.leadingAnchor.constraint(equalTo: cell.contentView.layoutMarginsGuide.leadingAnchor),
                        textField.trailingAnchor.constraint(equalTo: cell.contentView.layoutMarginsGuide.trailingAnchor),
                    ])
                }

                cell.selectionStyle = .none
            default: break
        }
        return cell
    }
}
