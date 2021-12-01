package com.FeelRing.utils;

import android.app.AlertDialog;
import android.app.Dialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;

public class PopupUtil {
    private static ProgressDialog sProgress;

    public static Dialog showPopup(Context context,
                                   String title, String message,
                                   String confirm, String cancel,
                                   BFunction confirmFunction,
                                   BFunction cancelBFunction) {

        AlertDialog.Builder builder = new AlertDialog.Builder(context);

        builder.setTitle(title);
        builder.setMessage(message);
        builder.setCancelable(false);
        builder.setPositiveButton(confirm, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (confirmFunction != null)
                    confirmFunction.run();
            }
        });
        builder.setNegativeButton(cancel, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (cancelBFunction != null)
                    cancelBFunction.run();
                dialog.dismiss();
            }
        });

        return builder.create();

    }

    public static void showProgress(Context context,
                                    int style,
                                    String title, String message) {
        hideProgress();

        sProgress = new ProgressDialog(context);
        sProgress.setProgressStyle(style);

        if (null != title) {
            sProgress.setTitle(title);
        }
        sProgress.setMessage(message);
        sProgress.setCancelable(false);

        sProgress.show();

    }

    public static void hideProgress() {
        if (null == sProgress)
            return;
        sProgress.dismiss();
        sProgress = null;
    }
}
