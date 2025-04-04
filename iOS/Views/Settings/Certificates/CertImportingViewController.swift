// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit
import UniformTypeIdentifiers

// External C functions from openssl_tools
@_silgen_name("provision_file_validation")
func provision_file_validation(_ path: String)

@_silgen_name("p12_password_check")
func p12_password_check(_ path: String, _ password: String) -> Bool

class CertImportingViewController: UITableViewController {
    lazy var saveButton = UIBarButtonItem(title: String.localized("SAVE"), style: .plain, target: self, action: #selector(saveAction))
    private var passwordTextField: UITextField?

    enum FileType: Hashable {
        case provision
        case p12
        case password
    }

    var sectionData = [
        "provision",
        "certs",
        "pass",
    ]

    private var selectedFiles: [FileType: Any] = [:]

    init() { super.init(style: .insetGrouped) }
    @available(*, unavailable)
    required init?(coder _: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupNavigation()
        setupViews()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
    }

    fileprivate func setupViews() {
        self.tableView.dataSource = self
        self.tableView.delegate = self
    }

    fileprivate func setupNavigation() {
        self.navigationItem.largeTitleDisplayMode = .never
        self.title = String.localized("CERT_IMPORTING_VIEWCONTROLLER_TITLE")
        saveButton.isEnabled = false
        self.navigationItem.rightBarButtonItem = saveButton
        self.navigationItem.leftBarButtonItem = UIBarButtonItem(title: String.localized("DISMISS"), style: .done, target: self, action: #selector(closeSheet))
    }

    @objc func closeSheet() {
        dismiss(animated: true, completion: nil)
    }

    @objc func saveAction() {
        // krilling myself, also improving code later just not today lmaoo
        let mobileProvisionPath: URL!
        mobileProvisionPath = (selectedFiles[.provision] as! URL)
        #if !targetEnvironment(simulator)
            if let p12path = selectedFiles[.p12] as? URL {
                // Call functions from openssl_tools.hpp
                provision_file_validation(mobileProvisionPath.path)
                if !p12_password_check(p12path.path, selectedFiles[.password] as? String ?? "") {
                    let alert = UIAlertController(title: String.localized("CERT_IMPORTING_VIEWCONTROLLER_PW_ALERT_TITLE"), message: String.localized("CERT_IMPORTING_VIEWCONTROLLER_PW_ALERT_DESCRIPTION"), preferredStyle: UIAlertController.Style.alert)
                    alert.addAction(UIAlertAction(title: String.localized("OK"), style: UIAlertAction.Style.default, handler: nil))
                    self.present(alert, animated: true, completion: nil)
                    return
                }
            }
        #endif

        if let fileContent = CertData.parseMobileProvisioningFile(atPath: mobileProvisionPath) {
            CoreDataManager.shared.addToCertificates(cert: fileContent, files: selectedFiles)
            self.dismiss(animated: true)
        } else {
            Debug.shared.log(message: String.localized("ERROR_FAILED_TO_READ_MOBILEPROVISION"), type: .error)
        }
    }

    @objc func textFieldDidChange(_ textField: UITextField) {
        guard textField === passwordTextField else { return }

        if let password = textField.text {
            selectedFiles[.password] = password
        }
    }
}

extension CertImportingViewController {
    override func numberOfSections(in _: UITableView) -> Int { return sectionData.count }
    override func tableView(_: UITableView, numberOfRowsInSection _: Int) -> Int { return 1 }

    override func tableView(_: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = UITableViewCell(style: .subtitle, reuseIdentifier: "Cell")
        cell.selectionStyle = .default

        let imageView = UIImageView(image: UIImage(systemName: "circle"))
        imageView.tintColor = .quaternaryLabel
        cell.accessoryView = imageView

        cell.textLabel?.font = .boldSystemFont(ofSize: 15)
        cell.detailTextLabel?.textColor = .secondaryLabel

        let fileType: FileType

        switch sectionData[indexPath.section] {
            case "provision":
                cell.textLabel?.text = String.localized("CERT_IMPORTING_VIEWCONTROLLER_CELL_IMPORT_PROV")
                cell.detailTextLabel?.text = ".mobileprovision"
                fileType = .provision
            case "certs":
                cell.textLabel?.text = String.localized("CERT_IMPORTING_VIEWCONTROLLER_CELL_IMPORT_CERT")
                cell.detailTextLabel?.text = ".p12"

                if selectedFiles[.p12] != nil {
                    let checkmarkImage = UIImage(systemName: "checkmark")
                    let checkmarkImageView = UIImageView(image: checkmarkImage)
                    checkmarkImageView.tintColor = .systemBlue
                    cell.accessoryView = checkmarkImageView
                } else {
                    let circleImage = UIImage(systemName: "circle")
                    let circleImageView = UIImageView(image: circleImage)
                    circleImageView.tintColor = .quaternaryLabel
                    cell.accessoryView = circleImageView
                }

                return cell
            case "pass":
                let passwordCell = UITableViewCell(style: .default, reuseIdentifier: "PasswordCell")
                let textField = UITextField(frame: CGRect(x: 0, y: 0, width: 200, height: 30))

                textField.placeholder = String.localized("CERT_IMPORTING_VIEWCONTROLLER_CELL_IMPORT_ENTER_PW")
                textField.isSecureTextEntry = true
                textField.addTarget(self, action: #selector(textFieldDidChange(_:)), for: .editingChanged)

                passwordCell.textLabel?.text = String.localized("CERT_IMPORTING_VIEWCONTROLLER_CELL_IMPORT_PW")
                passwordCell.selectionStyle = .none
                passwordCell.accessoryView = textField

                passwordTextField = textField

                return passwordCell
            default:
                return cell
        }

        if selectedFiles[fileType] != nil {
            let checkmarkImage = UIImage(systemName: "checkmark")
            let checkmarkImageView = UIImageView(image: checkmarkImage)
            checkmarkImageView.tintColor = .systemBlue
            cell.accessoryView = checkmarkImageView
        } else {
            let circleImage = UIImage(systemName: "circle")
            let circleImageView = UIImageView(image: circleImage)
            circleImageView.tintColor = .quaternaryLabel
            cell.accessoryView = circleImageView
        }

        return cell
    }

    override func tableView(_: UITableView, titleForFooterInSection section: Int) -> String? {
        switch sectionData[section] {
            case "provision":
                return String.localized("CERT_IMPORTING_VIEWCONTROLLER_FOOTER_PROV")
            case "certs":
                return String.localized("CERT_IMPORTING_VIEWCONTROLLER_FOOTER_CERT")
            case "pass":
                return String.localized("CERT_IMPORTING_VIEWCONTROLLER_FOOTER_PASS")
            default:
                return nil
        }
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let fileType: FileType

        switch sectionData[indexPath.section] {
            case "provision":
                fileType = .provision
            case "certs":
                fileType = .p12
            default:
                return
        }

        guard selectedFiles[fileType] == nil else {
            tableView.deselectRow(at: indexPath, animated: true)
            return
        }

        switch sectionData[indexPath.section] {
            case "provision":
                presentDocumentPicker(fileExtension: [UTType(filenameExtension: "mobileprovision")!])
            case "certs":
                presentDocumentPicker(fileExtension: [UTType(filenameExtension: "p12")!])
            default:
                return
        }

        tableView.deselectRow(at: indexPath, animated: true)
    }
}

extension CertImportingViewController: UIDocumentPickerDelegate {
    func presentDocumentPicker(fileExtension: [UTType]) {
        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: fileExtension, asCopy: true)
        documentPicker.delegate = self
        documentPicker.allowsMultipleSelection = false
        present(documentPicker, animated: true, completion: nil)
    }

    func documentPicker(_: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let selectedFileURL = urls.first else { return }

        let fileType: FileType?

        switch selectedFileURL.pathExtension {
            case "mobileprovision":
                fileType = .provision
            case "p12":
                fileType = .p12
            default:
                fileType = nil
        }

        if let fileType = fileType {
            selectedFiles[fileType] = selectedFileURL
            tableView.reloadData()
        }

        if (selectedFiles[.provision] != nil) && (selectedFiles[.p12] != nil) {
            saveButton.isEnabled = true
        }
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        controller.dismiss(animated: true, completion: nil)
    }

    func checkIfFileIsCert(cert: URL?) -> Bool {
        guard let cert = cert, cert.isFileURL else { return false }

        do {
            let fileContent = try String(contentsOf: cert, encoding: .utf8)
            if fileContent.contains("BEGIN CERTIFICATE") {
                return true
            }
        } catch {
            Debug.shared.log(message: "Error reading file: \(error)")
        }

        return false
    }
}
