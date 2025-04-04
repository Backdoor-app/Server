// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

class TweakLibraryViewCell: UITableViewCell {
    public var segmentedControl: UISegmentedControl = {
        let control = UISegmentedControl()
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        contentView.addSubview(segmentedControl)
        setupConstraints()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        contentView.addSubview(segmentedControl)
        setupConstraints()
    }

    private func setupConstraints() {
        NSLayoutConstraint.activate([
            segmentedControl.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            segmentedControl.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 15),
            segmentedControl.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -15),
        ])
    }

    func configureSegmentedControl(with items: [String], selectedIndex: Int) {
        segmentedControl.removeAllSegments()
        for (index, item) in items.enumerated() {
            segmentedControl.insertSegment(withTitle: item, at: index, animated: false)
        }
        segmentedControl.selectedSegmentIndex = selectedIndex
    }
}
