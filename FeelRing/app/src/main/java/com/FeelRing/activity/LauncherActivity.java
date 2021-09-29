package com.FeelRing.activity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.widget.TextView;

import androidx.core.content.ContextCompat;

import com.FeelRing.R;
import com.FeelRing.utils.BFunction;
import com.FeelRing.utils.Const;
import com.FeelRing.utils.Const.CHECK_STATUS;
import com.FeelRing.utils.NetUtil;

public class LauncherActivity extends BaseActivity {

    TextView tvVersion;
    CHECK_STATUS STATUS = CHECK_STATUS.STATUS_NETWORK;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_launcher);

        initControls();
    }

    @Override
    protected void onResume() {
        super.onResume();
        checkEnvironments();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    private void initControls() {
        tvVersion = (TextView) findViewById(R.id.tv_version);
        tvVersion.setText(getAppVersionName());

    }

    private void checkEnvironments() {
        Log.d(Const.TAG, "check network == ");

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
                    STATUS = CHECK_STATUS.STATUS_NICKNAME;
                }
            }

//            case STATUS_PERMISSION: {
//                if (!checkPermissions()) {
//                    //ActivityCompat.requestPermissions(getActivity(), new String[] {Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
//                } else {
//                    STATUS = CHECK_STATUS.STATUS_NICKNAME;
//                }
//            }

            case STATUS_NICKNAME: {
                if (!checkNickName()) {
                    Intent intent = new Intent(getActivity(), SurveyActivity.class);
                    startActivity(intent);
                } else {
                    Intent intent = new Intent(getActivity(), MainActivity.class);
                    startActivity(intent);
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