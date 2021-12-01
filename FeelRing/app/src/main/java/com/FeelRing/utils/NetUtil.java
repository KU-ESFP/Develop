package com.FeelRing.utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

public class NetUtil {

    public static boolean isConnected(Context context) {
        try {
            ConnectivityManager connManager = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkInfo netInfo = connManager.getActiveNetworkInfo();

            if (connManager == null || netInfo == null) {
                return false;
            }

            return netInfo.isConnected();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

}
