package com.FeelRing.activity;

import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.widget.TextView;
import com.FeelRing.R;
import com.FeelRing.utils.BFunction;
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
        Log.d("rsj", "check network == ");

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
            Log.d("rsj", "Network Connected");
            return true;
        } else {
            Log.d("rsj","Network Unconnected");
            return false;
        }
    }

    // 2. 닉네임 있는 지 확인
    private boolean checkNickName() {
        if (getNickName().length() > 0) {
            Log.d("rsj", "Nick name == " + getNickName());
            return true;
        } else {
            Log.d("rsj", "Nick name is not exist");
            return false;
        }
    }


    // 2. 카메라 권한 받기



}