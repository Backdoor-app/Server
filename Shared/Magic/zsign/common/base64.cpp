/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted
 * under the terms of the Proprietary Software License.
 */

#include "base64.h"
#include <string.h>

#define B0(a) (a & 0xFF)
#define B1(a) (a >> 8 & 0xFF)
#define B2(a) (a >> 16 & 0xFF)
#define B3(a) (a >> 24 & 0xFF)

ZBase64::ZBase64(void) {}

ZBase64::~ZBase64(void) {
    if (!m_arrEnc.empty()) {
        for (size_t i = 0; i < m_arrEnc.size(); i++) {
            delete[] m_arrEnc[i];
        }
        m_arrEnc.clear();
    }

    if (!m_arrDec.empty()) {
        for (size_t i = 0; i < m_arrDec.size(); i++) {
            delete[] m_arrDec[i];
        }
        m_arrDec.clear();
    }
}

unsigned char ZBase64::s_ca_table_enc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Implementation of encoding functions
char* ZBase64::Encode(const char* pData, int nDataLen) {
    if (nullptr == pData || nDataLen <= 0) {
        return nullptr;
    }
    
    int nEncodedLen = (nDataLen + 2) / 3 * 4;
    char* pEncoded = new char[nEncodedLen + 1];
    m_arrEnc.push_back(pEncoded);
    
    int nRemain = nDataLen % 3;
    int nLoopTime = nDataLen / 3;
    
    const unsigned char* p = (const unsigned char*)pData;
    char* q = pEncoded;
    
    for (int i = 0; i < nLoopTime; i++) {
        q[0] = s_ca_table_enc[p[0] >> 2];
        q[1] = s_ca_table_enc[((p[0] & 0x03) << 4) | (p[1] >> 4)];
        q[2] = s_ca_table_enc[((p[1] & 0x0f) << 2) | (p[2] >> 6)];
        q[3] = s_ca_table_enc[p[2] & 0x3f];
        
        p += 3;
        q += 4;
    }
    
    if (0 < nRemain) {
        q[0] = s_ca_table_enc[p[0] >> 2];
        if (1 == nRemain) {
            q[1] = s_ca_table_enc[(p[0] & 0x03) << 4];
            q[2] = '=';
            q[3] = '=';
        } else {
            q[1] = s_ca_table_enc[((p[0] & 0x03) << 4) | (p[1] >> 4)];
            q[2] = s_ca_table_enc[(p[1] & 0x0f) << 2];
            q[3] = '=';
        }
        q += 4;
    }
    
    *q = '\0';
    return pEncoded;
}

char* ZBase64::Encode(const string& strData) {
    return Encode(strData.c_str(), (int)strData.size());
}

// Implementation of decoding functions
const char* ZBase64::Decode(const char* pData, int nDataLen, int* pOutDataLen) {
    if (nullptr == pData || nDataLen <= 0) {
        return nullptr;
    }
    
    // Skip whitespace and get actual length
    int nRealLen = 0;
    for (int i = 0; i < nDataLen; i++) {
        if (!isspace((unsigned char)pData[i])) {
            nRealLen++;
        }
    }
    
    // Length check
    if (0 != (nRealLen % 4)) {
        return nullptr;
    }
    
    int nPadding = 0;
    if (nRealLen > 0) {
        if ('=' == pData[nDataLen - 1]) nPadding++;
        if ('=' == pData[nDataLen - 2]) nPadding++;
    }
    
    *pOutDataLen = nRealLen / 4 * 3 - nPadding;
    char* pDecoded = new char[*pOutDataLen + 1];
    m_arrDec.push_back(pDecoded);
    
    static unsigned char s_ca_table_dec[256] = {0};
    static bool s_b_init = false;
    if (!s_b_init) {
        memset(s_ca_table_dec, 0xff, sizeof(s_ca_table_dec));
        for (int i = 0; i < 64; i++) {
            s_ca_table_dec[s_ca_table_enc[i]] = i;
        }
        s_b_init = true;
    }
    
    const unsigned char* p = (const unsigned char*)pData;
    unsigned char* q = (unsigned char*)pDecoded;
    int j = 0;
    
    for (int i = 0; i < nDataLen; i++) {
        if (isspace(p[i])) {
            continue;
        }
        
        unsigned char c = s_ca_table_dec[p[i]];
        if (0xff == c) {
            continue;
        }
        
        switch (j % 4) {
            case 0:
                q[0] = c << 2;
                break;
            case 1:
                q[0] |= c >> 4;
                q[1] = c << 4;
                break;
            case 2:
                q[1] |= c >> 2;
                q[2] = c << 6;
                break;
            case 3:
                q[2] |= c;
                q += 3;
                break;
        }
        j++;
    }
    
    pDecoded[*pOutDataLen] = '\0';
    return pDecoded;
}

const char* ZBase64::Decode(const string& strData, int* pOutDataLen) {
    return Decode(strData.c_str(), (int)strData.size(), pOutDataLen);
}
