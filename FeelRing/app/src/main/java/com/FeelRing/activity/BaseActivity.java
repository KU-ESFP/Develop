package com.FeelRing.activity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.FeelRing.R;
import com.FeelRing.datebase.DBOpenHelper;
import com.FeelRing.utils.BFunction;
import com.FeelRing.utils.Const;
import com.FeelRing.utils.PopupUtil;
import com.FeelRing.utils.ToastUtil;

import java.io.File;

public class BaseActivity extends AppCompatActivity {
    final String activityName = "::AnalysisActivity";

    public static DBOpenHelper dbHelper;
    public Uri photoURI;
    public File photoFile;
    public String photoPath;
    public long backKeyPressedTime = 0;

    public BaseActivity getActivity() {
        return this;
    }

    public String getAppVersionName() {
        try {
            return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public int getAppVersionCode() {
        try {
            return getPackageManager().getPackageInfo(getPackageName(), 0).versionCode;
        } catch (PackageManager.NameNotFoundException e) {
            return 0;
        }
    }

    public void showPopUp(String title, BFunction okBF, BFunction cancelBF) {
        showPopUp(title, null, "확인", "취소", okBF, cancelBF);
    }

    public void showPopUp(int title, BFunction okBF, BFunction cancelBF) {
        showPopUp(getString(title), null, "확인", "취소", okBF, cancelBF);
    }

    public void showPopUp(String title, String message, String ok, String cancel, BFunction okBFunction, BFunction cancelBFunction) {
        PopupUtil.showPopup(this, title, message, ok, cancel, okBFunction, cancelBFunction).show();
    }

    public void showToast(int msg) {
        showToast(getString(msg));
    }

    public void showToast(String msg) {
        ToastUtil.showToast(this, msg);
    }

    public void hideToast() {
        ToastUtil.hideToast();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    public boolean emptyDBTable() {
        if (dbHelper == null) return true;

        if (dbHelper.getCountRecord() == 0) {
            Log.d(Const.TAG + activityName, "empty");
            return true;
        }

        Log.d(Const.TAG, "NOT empty");
        return false;
    }

    public String getNameColumn() {
        String name = "";

        if (dbHelper != null) {
            Cursor cr = dbHelper.selectColumns();

            while (cr.moveToNext()) {
                name = cr.getString(cr.getColumnIndex("name"));
            }
        }

        return name;
    }

    public void exitProgram() {
        moveTaskToBack(true); // 태스크를 백그라운드로 이동

        if (Build.VERSION.SDK_INT >= 21) {
            // 액티비티 종료 + 태스크 리스트에서 지우기
            finishAndRemoveTask();
        } else {
            // 액티비티 종료
            finish();
        }

        System.exit(0);
    }

    @Override
    public void onBackPressed() {
        // super.onBackPressed();

        if (System.currentTimeMillis() > backKeyPressedTime + 2500) {
            backKeyPressedTime = System.currentTimeMillis();
            showToast(R.string.touch_one_backpress);
            Log.d(Const.TAG + activityName, "back press time == " + backKeyPressedTime);
            Log.d(Const.TAG + activityName, "system time" + System.currentTimeMillis());
            return;
        }
        // 마지막으로 뒤로 가기 버튼을 눌렀던 시간에 2.5초를 더해 현재 시간과 비교 후
        // 마지막으로 뒤로 가기 버튼을 눌렀던 시간이 2.5초가 지나지 않았으면 종료
        if (System.currentTimeMillis() <= backKeyPressedTime + 2500) {
            Log.d(Const.TAG + activityName, "finish!!");
            if (dbHelper != null) dbHelper.close();

            Intent intent = new Intent(getActivity(), LauncherActivity.class);
            startActivity(intent);
            exitProgram();
        }
    }

}