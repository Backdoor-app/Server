// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class AddIdentifierViewController: UITableViewController {
    var onAdd: ((String, String) -> Void)?

    private let identifierTextField: UITextField = {
        let textField = UITextField(frame: .zero)
        textField.placeholder = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_IDENTIFIERS_ID")
        textField.borderStyle = .none
        textField.translatesAutoresizingMaskIntoConstraints = false
        return textField
    }()

    private let replacementTextField: UITextField = {
        let textField = UITextField(frame: .zero)
        textField.placeholder = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_IDENTIFIERS_ID_REPLACEMENT")
        textField.borderStyle = .none
        textField.translatesAutoresizingMaskIntoConstraints = false
        return textField
    }()

    init(mode: IdentifierMode) {
        super.init(style: .insetGrouped)

        switch mode {
            case .bundleId:
                identifierTextField.placeholder = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_IDENTIFIERS_ID")
                replacementTextField.placeholder = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_IDENTIFIERS_ID_REPLACEMENT")
            case .displayName:
                identifierTextField.placeholder = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_DISPLAYNAMES_ID")
                replacementTextField.placeholder = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_DISPLAYNAMES_ID_REPLACEMENT")
        }
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = String.localized("APP_SIGNING_VIEW_CONTROLLER_CELL_SIGNING_OPTIONS_IDENTIFIERS_NEW")
        view.backgroundColor = .systemBackground

        identifierTextField.addTarget(self, action: #selector(textFieldsDidChange), for: .editingChanged)
        replacementTextField.addTarget(self, action: #selector(textFieldsDidChange), for: .editingChanged)

        navigationItem.rightBarButtonItem = UIBarButtonItem(
            title: String.localized("ADD"),
            style: .done,
            target: self,
            action: #selector(addButtonTapped)
        )
        navigationItem.rightBarButtonItem?.isEnabled = false
    }

    override func numberOfSections(in _: UITableView) -> Int {
        return 1
    }

    override func tableView(_: UITableView, numberOfRowsInSection _: Int) -> Int {
        return 2
    }

    override func tableView(_: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = UITableViewCell(style: .default, reuseIdentifier: nil)
        cell.selectionStyle = .none

        switch indexPath.row {
            case 0:
                if identifierTextField.superview == nil {
                    cell.contentView.addSubview(identifierTextField)
                    NSLayoutConstraint.activate([
                        identifierTextField.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
                        identifierTextField.leadingAnchor.constraint(equalTo: cell.contentView.layoutMarginsGuide.leadingAnchor),
                        identifierTextField.trailingAnchor.constraint(equalTo: cell.contentView.layoutMarginsGuide.trailingAnchor),
                    ])
                }

            case 1:
                if replacementTextField.superview == nil {
                    cell.contentView.addSubview(replacementTextField)
                    NSLayoutConstraint.activate([
                        replacementTextField.centerYAnchor.constraint(equalTo: cell.contentView.centerYAnchor),
                        replacementTextField.leadingAnchor.constraint(equalTo: cell.contentView.layoutMarginsGuide.leadingAnchor),
                        replacementTextField.trailingAnchor.constraint(equalTo: cell.contentView.layoutMarginsGuide.trailingAnchor),
                    ])
                }

            default:
                return cell
        }

        return cell
    }

    @objc private func textFieldsDidChange() {
        let identifierText = identifierTextField.text ?? ""
        let replacementText = replacementTextField.text ?? ""
        navigationItem.rightBarButtonItem?.isEnabled = !identifierText.isEmpty && !replacementText.isEmpty
    }

    @objc private func addButtonTapped() {
        if let identifier = identifierTextField.text,
           let replacement = replacementTextField.text,
           !identifier.isEmpty, !replacement.isEmpty
        {
            onAdd?(identifier, replacement)
            navigationController?.popViewController(animated: true)
        }
    }
}
