package com.FeelRing.activity;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.FeelRing.R;
import com.FeelRing.utils.Const;

public class SurveyActivity extends BaseActivity {
    EditText etInputName;
    Button btNext;

    private long backKeyPressedTime = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_survey);

        initControls();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    private void initControls() {
        etInputName = (EditText) findViewById(R.id.et_input_name);
        btNext = (Button) findViewById(R.id.bt_next);

        btNext.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO: 어플 종료하면 닉네임 정보 사라짐 => SQLite에 저장하도록 구현하기
                String nickName = String.valueOf(etInputName.getText());
                setNickName(nickName);

                if (!checkNickName()) {
                    showToast(R.string.nickname_not_exist);
                    etInputName.setHintTextColor(getResources().getColor(R.color.red));
                } else {
                    Intent intent = new Intent(getActivity(), MainActivity.class);
                    startActivity(intent);
                }
            }
        });


    }

    private boolean checkNickName() {
        if (getNickName().length() > 0) {
            Log.d(Const.TAG, "Nick name == " + getNickName());
            return true;
        } else {
            Log.d(Const.TAG, "Nick name is not exist");
            return false;
        }
    }

    private void exitProgram() {
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
            Log.d(Const.TAG, "aaa back press time == " + backKeyPressedTime);
            Log.d(Const.TAG, "system time" + System.currentTimeMillis());
            return;
        }
        // 마지막으로 뒤로 가기 버튼을 눌렀던 시간에 2.5초를 더해 현재 시간과 비교 후
        // 마지막으로 뒤로 가기 버튼을 눌렀던 시간이 2.5초가 지나지 않았으면 종료
        if (System.currentTimeMillis() <= backKeyPressedTime + 2500) {
            Log.d(Const.TAG, "finish!!");

            exitProgram();
        }
    }


}