package com.FeelRing.activity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.graduate.FeelRing.R;

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

        String nickName = String.valueOf(etInputName.getText());
        setNickName(nickName);

        btNext.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
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
            Log.d("rsj", "Nick name == " + getNickName());
            return true;
        } else {
            Log.d("rsj", "Nick name is not exist");
            return false;
        }
    }


}