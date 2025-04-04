/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted
 * under the terms of the Proprietary Software License.
 */

/*
 */

#ifndef zsign_hpp
#define zsign_hpp

#import <Foundation/Foundation.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

bool InjectDyLib(NSString *filePath, NSString *dylibPath, bool weakInject, bool bCreate);

bool ChangeDylibPath(NSString *filePath, NSString *oldPath, NSString *newPath);

bool ListDylibs(NSString *filePath, NSMutableArray *dylibPathsArray);
bool UninstallDylibs(NSString *filePath, NSArray<NSString *> *dylibPathsArray);

int zsign(NSString *app, NSString *prov, NSString *key, NSString *pass, NSString *bundleid, NSString *displayname,
          NSString *bundleversion, bool dontGenerateEmbeddedMobileProvision);

#ifdef __cplusplus
}
#endif

#endif /* zsign_hpp */
