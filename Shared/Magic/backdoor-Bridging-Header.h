/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted
 * under the terms of the Proprietary Software License.
 */

// Import Foundation and UIKit frameworks
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

// Import private extensions
#import "LSApplicationWorkspace.h"
#import "UISheetPresentationControllerDetent+Private.h"

// Import C++ headers
#ifdef __cplusplus
#import "openssl_tools.hpp"
#import "zsign.hpp"
#endif
