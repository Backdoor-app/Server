// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

class SwitchViewCell: UITableViewCell {
    let switchControl: UISwitch = {
        let switchControl = UISwitch()
        switchControl.translatesAutoresizingMaskIntoConstraints = false
        return switchControl
    }()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        contentView.addSubview(switchControl)
        setupConstraints()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        contentView.addSubview(switchControl)
        setupConstraints()
    }

    private func setupConstraints() {
        NSLayoutConstraint.activate([
            switchControl.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            switchControl.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -22),
        ])
    }
}
