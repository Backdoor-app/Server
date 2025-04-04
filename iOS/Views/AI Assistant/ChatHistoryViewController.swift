// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class ChatHistoryViewController: UITableViewController {
    private var sessions: [ChatSession] = []
    var didSelectSession: ((ChatSession) -> Void)?

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Chat History"
        navigationItem.rightBarButtonItem = UIBarButtonItem(barButtonSystemItem: .done, target: self, action: #selector(dismissVC))
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "Cell")
        tableView.backgroundColor = .systemGray6
        loadSessions()
    }

    @objc private func dismissVC() {
        dismiss(animated: true)
    }

    private func loadSessions() {
        sessions = CoreDataManager.shared.getChatSessions()
        tableView.reloadData()
    }

    override func tableView(_: UITableView, numberOfRowsInSection _: Int) -> Int {
        return sessions.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath)
        let session = sessions[indexPath.row]
        cell.textLabel?.text = session.title
        cell.backgroundColor = .systemBackground
        cell.accessoryType = .disclosureIndicator
        return cell
    }

    override func tableView(_: UITableView, didSelectRowAt indexPath: IndexPath) {
        let session = sessions[indexPath.row]
        didSelectSession?(session)
        dismiss(animated: true)
    }
}
