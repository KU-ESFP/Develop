package com.FeelRing.activity;

import android.content.pm.PackageManager;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.FeelRing.R;
import com.FeelRing.utils.BFunction;
import com.FeelRing.utils.Const;
import com.FeelRing.utils.PopupUtil;
import com.FeelRing.utils.ToastUtil;

public class BaseActivity extends AppCompatActivity {
    private String nickName = "";
    private long backKeyPressedTime = 0;

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

    @Override
    public void onBackPressed() {
        // super.onBackPressed();

        if (System.currentTimeMillis() > backKeyPressedTime + 2500) {
            backKeyPressedTime = System.currentTimeMillis();
            showToast(R.string.touch_one_backpress);
            Log.d(Const.TAG, "back press time == " + backKeyPressedTime);
            Log.d(Const.TAG, "system time" + System.currentTimeMillis());
            return;
        }
        // 마지막으로 뒤로 가기 버튼을 눌렀던 시간에 2.5초를 더해 현재 시간과 비교 후
        // 마지막으로 뒤로 가기 버튼을 눌렀던 시간이 2.5초가 지나지 않았으면 종료
        if (System.currentTimeMillis() <= backKeyPressedTime + 2500) {
            Log.d(Const.TAG, "finish!!");

            // TODO: 로그 찍히는거로 봐서 조건문은 들어오는데 finish()가 제대로 되지 않음..해결하기
            finish();
        }
    }
}