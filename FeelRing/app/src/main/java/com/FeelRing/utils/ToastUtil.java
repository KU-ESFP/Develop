package com.FeelRing.utils;

import android.content.Context;
import android.widget.Toast;

public class ToastUtil {
    private static Toast toast;

    public static void showToast(Context context, String msg) {
        hideToast();

        toast = Toast.makeText(context, msg, Toast.LENGTH_SHORT);

        toast.show();
    }

    public static void hideToast() {
        if (toast == null)
            return;
        toast.cancel();
        toast = null;
    }
}
