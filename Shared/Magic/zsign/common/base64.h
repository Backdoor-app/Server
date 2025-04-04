/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted
 * under the terms of the Proprietary Software License.
 */

#pragma once

#include <string>
#include <vector>
using namespace std;

class ZBase64 {
public:
    ZBase64(void);
    ~ZBase64(void);

    char *Encode(const char *pData, int nDataLen);
    char *Encode(const string &strData);
    const char *Decode(const char *pData, int nDataLen, int *pOutDataLen);
    const char *Decode(const string &strData, int *pOutDataLen);

private:
    vector<char *> m_arrEnc;
    vector<char *> m_arrDec;
    static unsigned char s_ca_table_enc[];
};
