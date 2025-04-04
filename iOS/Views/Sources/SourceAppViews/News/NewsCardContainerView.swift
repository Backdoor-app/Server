// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import SwiftUI

struct NewsCardContainerView: View {
    @Binding var isSheetPresented: Bool
    var news: NewsData
    @Namespace private var namespace

    let uuid = UUID().uuidString

    var body: some View {
        Button(action: {
            isSheetPresented = true
        }) {
            NewsCardView(news: news)
                .fullScreenCover(isPresented: $isSheetPresented) {
                    CardContextMenuView(news: news)
                        .compatNavigationTransition(id: uuid, ns: namespace)
                }
                .compatMatchedTransitionSource(id: uuid, ns: namespace)
                .compactContentMenuPreview(news: news)
        }
    }
}

extension View {
    func compactContentMenuPreview(news: NewsData) -> some View {
        if #available(iOS 16.0, *) {
            return self.contextMenu {
                if news.url != nil {
                    Button(action: {
                        UIApplication.shared.open(news.url!)
                    }) {
                        Label("Open URL", systemImage: "arrow.up.right")
                    }
                }
            }
        } else {
            return self
        }
    }
}
