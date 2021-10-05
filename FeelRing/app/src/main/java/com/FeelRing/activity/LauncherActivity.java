package com.FeelRing.activity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.widget.TextView;

import androidx.core.content.ContextCompat;

import com.FeelRing.R;
import com.FeelRing.datebase.DBOpenHelper;
import com.FeelRing.utils.BFunction;
import com.FeelRing.utils.Const;
import com.FeelRing.utils.Const.CHECK_STATUS;
import com.FeelRing.utils.NetUtil;

public class LauncherActivity extends BaseActivity {
    final String activityName = "::LauncherActivity";

    TextView tvVersion;
    CHECK_STATUS STATUS = CHECK_STATUS.STATUS_NETWORK;
    Context context;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_launcher);
        context = LauncherActivity.this;

        dbControls();
        initControls();
    }

    @Override
    protected void onResume() {
        super.onResume();
        checkEnvironments();
    }

    private void dbControls() {
        dbHelper = new DBOpenHelper(context);
        dbHelper.open();
        dbHelper.create();
    }

    private void initControls() {
        tvVersion = (TextView) findViewById(R.id.tv_version);
        tvVersion.setText(getAppVersionName());

    }

    private void checkEnvironments() {
        switch (STATUS) {
            case STATUS_NETWORK: {

                if (!checkNetwork()) {
                    showPopUp(R.string.network_not_connected, new BFunction() {
                        @Override
                        public void run() {
                            startActivityForResult(new Intent(Settings.ACTION_WIFI_SETTINGS), 0);
                        }
                    }, new BFunction() {
                        @Override
                        public void run() {
                            finish();
                        }
                    });
                    return;
                } else {
                    STATUS = CHECK_STATUS.STATUS_NAME;
                }
            }
            // 이름 이미 저장되어있으면 서베이 액티비티 건너뛰고 바로 메인액티비티로
            case STATUS_NAME: {
                if(!emptyDBTable()) {
                    Log.d(Const.TAG + activityName, "table is not null :: name = " + getNameColumn() + ":: go to main");
                    startActivity(new Intent(getActivity(), MainActivity.class));
                } else {
                    Log.d(Const.TAG + activityName, "table is null :: go to survey");
                    startActivity(new Intent(getActivity(), SurveyActivity.class));
                }
            }
        }
    }

    // 1. 인터넷 연결 확인
    private boolean checkNetwork() {
        if (NetUtil.isConnected(this)) {
            Log.d(Const.TAG, "Network Connected");
            return true;
        } else {
            Log.d(Const.TAG,"Network Unconnected");
            return false;
        }
    }

    // 2. 권한 확인
    private boolean checkPermissions() {
        if (ContextCompat.checkSelfPermission(getActivity(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(getActivity(), Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(getActivity(), Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            return false;
        }

        return true;
    }

    // 3. 닉네임 있는 지 확인
    private boolean checkNickName() {
        if (getNickName().length() > 0) {
            Log.d(Const.TAG, "Nick name == " + getNickName());
            return true;
        } else {
            Log.d(Const.TAG, "Nick name is not exist");
            return false;
        }
    }



}