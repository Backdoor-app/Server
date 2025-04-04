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

#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif
bool p12_password_check(NSString *file, NSString *pass);
void provision_file_validation(NSString *path);
void generate_root_ca_pair(const char *basename);
#ifdef __cplusplus
}
#endif
