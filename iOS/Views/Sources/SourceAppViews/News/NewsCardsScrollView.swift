// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import SwiftUI

struct NewsCardsScrollView: View {
    @State private var newsData: [NewsData]
    @State private var sheetStates: [String: Bool] = [:]
    @State var isSheetPresented = false

    init(newsData: [NewsData]) {
        _newsData = State(initialValue: newsData)
        print(newsData)
    }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(newsData.reversed(), id: \.self) { new in
                    let binding = Binding(
                        get: { sheetStates[new.identifier] ?? false },
                        set: { sheetStates[new.identifier] = $0 }
                    )

                    NewsCardContainerView(isSheetPresented: binding, news: new)
                }
            }
            .padding()
        }
        .frame(maxWidth: .infinity)
    }
}
