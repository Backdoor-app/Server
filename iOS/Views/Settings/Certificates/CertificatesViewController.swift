// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import UIKit

class CertificatesViewController: UITableViewController {
    var certs: [Certificate]?

    init() { super.init(style: .insetGrouped) }
    @available(*, unavailable)
    required init?(coder _: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupViews()
        NotificationCenter.default.addObserver(self, selector: #selector(afetch), name: Notification.Name("cfetch"), object: nil)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        setupNavigation()
        fetchSources()
    }

    deinit {
        NotificationCenter.default.removeObserver(self, name: Notification.Name("cfetch"), object: nil)
    }

    fileprivate func setupViews() {
        self.tableView.dataSource = self
        self.tableView.delegate = self
        self.tableView.tableHeaderView = UIView()
        self.tableView.register(UITableViewCell.self, forCellReuseIdentifier: "Cell")
        self.tableView.register(CertificateViewTableViewCell.self, forCellReuseIdentifier: "CertificateCell")
        self.tableView.register(CertificateViewAddTableViewCell.self, forCellReuseIdentifier: "AddCell")
    }

    fileprivate func setupNavigation() {
        self.title = String.localized("CERTIFICATES_VIEW_CONTROLLER_TITLE")
        self.navigationController?.navigationBar.prefersLargeTitles = false
    }

    @objc func addCert() {
        let viewController = CertImportingViewController()
        let navigationController = UINavigationController(rootViewController: viewController)

        if #available(iOS 15.0, *) {
            if let presentationController = navigationController.presentationController as? UISheetPresentationController {
                presentationController.detents = [.medium(), .large()]
            }
        }

        self.present(navigationController, animated: true)
    }
}

extension CertificatesViewController {
    override func numberOfSections(in _: UITableView) -> Int {
        return 2
    }

    override func tableView(_: UITableView, heightForHeaderInSection section: Int) -> CGFloat {
        switch section {
            case 0: return 40
            default: return 0
        }
    }

    override func tableView(_: UITableView, viewForHeaderInSection section: Int) -> UIView? {
        var title = ""

        switch section {
            case 0: title = String.localized("SETTINGS_VIEW_CONTROLLER_CELL_ADD_CERTIFICATES")
            default: break
        }

        let headerView = InsetGroupedSectionHeader(title: title)
        return headerView
    }

    override func tableView(_: UITableView, numberOfRowsInSection section: Int) -> Int {
        switch section {
            case 0: return 1
            case 1: return certs?.count ?? 0
            default: return 0
        }
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let reuseIdentifier = "Cell"
        let cell = UITableViewCell(style: .value1, reuseIdentifier: reuseIdentifier)

        switch indexPath.section {
            case 0:
                let cell = tableView.dequeueReusableCell(withIdentifier: "AddCell", for: indexPath) as! CertificateViewAddTableViewCell
                cell.configure(with: "plus")
                cell.selectionStyle = .none
                return cell

            case 1:
                let cell = tableView.dequeueReusableCell(withIdentifier: "CertificateCell", for: indexPath) as! CertificateViewTableViewCell
                let certificate = certs![indexPath.row]

                cell.configure(
                    with: certificate,
                    isSelected: Preferences.selectedCert == indexPath.row
                )

                return cell

            default:
                break
        }

        return cell
    }

    override func tableView(_ tableView: UITableView, contextMenuConfigurationForRowAt indexPath: IndexPath, point _: CGPoint) -> UIContextMenuConfiguration? {
        switch indexPath.section {
            case 1:
                let source = certs![indexPath.row]

                let configuration = UIContextMenuConfiguration(identifier: nil, actionProvider: { _ in
                    UIMenu(title: "", image: nil, identifier: nil, options: [], children: [
                        UIAction(title: String.localized("DELETE"), image: UIImage(systemName: "trash"), attributes: .destructive, handler: { _ in
                            if Preferences.selectedCert != indexPath.row {
                                do {
                                    CoreDataManager.shared.deleteAllCertificateContent(for: source)
                                    self.certs?.remove(at: indexPath.row)
                                    tableView.deleteRows(at: [indexPath], with: .automatic)
                                }
                            } else {
                                DispatchQueue.main.async {
                                    let alert = UIAlertController(title: String.localized("CERTIFICATES_VIEW_CONTROLLER_DELETE_ALERT_TITLE"), message: String.localized("CERTIFICATES_VIEW_CONTROLLER_DELETE_ALERT_DESCRIPTION"), preferredStyle: UIAlertController.Style.alert)
                                    alert.addAction(UIAlertAction(title: String.localized("LAME"), style: UIAlertAction.Style.default, handler: nil))
                                    self.present(alert, animated: true, completion: nil)
                                }
                            }
                        }),
                    ])
                })

                return configuration
            default:
                return nil
        }
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        switch indexPath.section {
            case 0:
                addCert()
            case 1:
                let previousSelectedCert = Preferences.selectedCert

                Preferences.selectedCert = indexPath.row

                var indexPathsToReload = [indexPath]
                if previousSelectedCert != indexPath.row {
                    indexPathsToReload.append(IndexPath(row: previousSelectedCert, section: 1))
                }

                tableView.reloadRows(at: indexPathsToReload, with: .fade)
                tableView.deselectRow(at: indexPath, animated: true)
                tableView.reloadSections(IndexSet([0]), with: .automatic)
            default:
                break
        }
    }
}

extension CertificatesViewController {
    @objc func afetch() {
        self.fetchSources()
    }

    func fetchSources() {
        do {
            self.certs = CoreDataManager.shared.getDatedCertificate()
            DispatchQueue.main.async {
                self.tableView.reloadData()
            }
        }
    }
}
