// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

func addAddButtonToView(title: String? = "+",
                        image: UIImage? = nil,
                        titleColor: UIColor = Preferences.appTintColor.uiColor,
                        backgroundColor: UIColor = UIColor(named: "Cells")!,
                        font: UIFont = UIFont.systemFont(ofSize: 20),
                        shadowOpacity: Float = 0.1,
                        shadowRadius: CGFloat = 11.0,
                        shadowOffset: CGSize = CGSize(width: 0, height: 0),
                        cornerRadius: CGFloat = 22.5,
                        cornerCurve: CALayerCornerCurve = .circular) -> UIButton
{
    let addButton = UIButton(type: .system)
    if let title = title {
        addButton.setTitle(title, for: .normal)
        addButton.setTitleColor(titleColor, for: .normal)
        addButton.titleLabel?.font = font
    } else if image != nil {
        addButton.setImage(UIImage(systemName: "folder.fill"), for: .normal)
        addButton.tintColor = Preferences.appTintColor.uiColor
    }
    addButton.backgroundColor = backgroundColor
    addButton.layer.shadowOpacity = shadowOpacity
    addButton.layer.shadowRadius = shadowRadius
    addButton.layer.shadowOffset = shadowOffset
    addButton.layer.cornerRadius = cornerRadius
    addButton.layer.cornerCurve = cornerCurve
    addButton.translatesAutoresizingMaskIntoConstraints = false

    return addButton
}
