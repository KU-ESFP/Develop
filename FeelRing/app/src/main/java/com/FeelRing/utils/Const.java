package com.FeelRing.utils;

public class Const {
    public enum CHECK_STATUS {
        STATUS_NETWORK,             // 네트워크 확인
        STATUS_NAME,
        STATUS_PERMISSION          // 권한 확인
    }

    //string
    public static String TAG                        = "rsj";

    // Request Codes
    public static int REQ_INTENT_CAMERA               = 1;
    public static int REQ_INTENT_GALLERY              = 2;

    // Type
    public static int CAMERA                          = 1001;
    public static int READ                            = 1002;
}
