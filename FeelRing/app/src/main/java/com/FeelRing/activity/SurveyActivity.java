package com.FeelRing.activity;

import android.content.Intent;
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


}