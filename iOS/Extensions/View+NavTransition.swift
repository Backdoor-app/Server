// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import SwiftUI

extension View {
    @ViewBuilder
    func compatNavigationTransition(id: String, ns: Namespace.ID) -> some View {
        if #available(iOS 18.0, *) {
            self.navigationTransition(.zoom(sourceID: id, in: ns))
        } else {
            self
        }
    }

    @ViewBuilder
    func compatMatchedTransitionSource(id: String, ns: Namespace.ID) -> some View {
        if #available(iOS 18.0, *) {
            self.matchedTransitionSource(id: id, in: ns)
        } else {
            self
        }
    }
}
