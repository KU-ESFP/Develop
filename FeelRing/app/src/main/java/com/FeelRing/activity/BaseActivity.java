package com.FeelRing.activity;

import androidx.appcompat.app.AppCompatActivity;

import android.content.pm.PackageManager;

import com.FeelRing.utils.BFunction;
import com.FeelRing.utils.PopupUtil;
import com.FeelRing.utils.ToastUtil;

public class BaseActivity extends AppCompatActivity {
    private String nickName = "";

    public BaseActivity getActivity() {
        return this;
    }

    public void setNickName(String nickName) {
        this.nickName = nickName;
    }

    public String getNickName() {
        return this.nickName;
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
}