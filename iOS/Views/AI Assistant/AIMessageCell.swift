// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class AIMessageCell: UITableViewCell {
    private let bubbleView = UIView()
    private let messageLabel = UILabel()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupViews()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupViews() {
        selectionStyle = .none
        backgroundColor = .clear

        bubbleView.backgroundColor = .systemGray5
        bubbleView.layer.cornerRadius = 16
        bubbleView.layer.maskedCorners = [.layerMaxXMinYCorner, .layerMaxXMaxYCorner, .layerMinXMaxYCorner]

        messageLabel.numberOfLines = 0
        messageLabel.textColor = .label
        messageLabel.font = .systemFont(ofSize: 16)

        contentView.addSubview(bubbleView)
        bubbleView.addSubview(messageLabel)

        bubbleView.translatesAutoresizingMaskIntoConstraints = false
        messageLabel.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            bubbleView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12),
            bubbleView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 6),
            bubbleView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -6),
            bubbleView.widthAnchor.constraint(lessThanOrEqualToConstant: 280),

            messageLabel.leadingAnchor.constraint(equalTo: bubbleView.leadingAnchor, constant: 12),
            messageLabel.trailingAnchor.constraint(equalTo: bubbleView.trailingAnchor, constant: -12),
            messageLabel.topAnchor.constraint(equalTo: bubbleView.topAnchor, constant: 8),
            messageLabel.bottomAnchor.constraint(equalTo: bubbleView.bottomAnchor, constant: -8),
        ])
    }

    func configure(with message: ChatMessage) {
        messageLabel.text = message.content
    }
}
