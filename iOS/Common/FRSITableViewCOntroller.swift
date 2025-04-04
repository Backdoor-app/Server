// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

class FRSITableViewCOntroller: FRSTableViewController {
    var signingDataWrapper: SigningDataWrapper
    var mainOptions: SigningMainDataWrapper

    init(signingDataWrapper: SigningDataWrapper, mainOptions: SigningMainDataWrapper) {
        self.signingDataWrapper = signingDataWrapper
        self.mainOptions = mainOptions

        super.init()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
