package com.FeelRing.activity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.FeelRing.R;
import com.FeelRing.utils.Const;

public class SurveyActivity extends BaseActivity {
    final String activityName = "::SurveyActivity";

    EditText etInputName;
    Button btNext;
    Context context;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_survey);
        context = SurveyActivity.this;

        //dbControls();
        initControls();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

//    private void dbControls() {
//        dbHelper = new DBOpenHelper(context);
//        dbHelper.open();
//        dbHelper.create();
//    }

//    private String getNameColumn() {
//        String name = "";
//        Cursor cr = dbHelper.selectColumns();
//
//        while (cr.moveToNext()) {
//            name = cr.getString(cr.getColumnIndex("name"));
//        }
//
//        return name;
//    }

//    private boolean emptyDBTable() {
//        if (dbHelper.getCountRecord() == 0) {
//            Log.d(Const.TAG, "empty");
//            return true;
//        }
//
//        Log.d(Const.TAG, "NOT empty");
//        return false;
//    }

    private void initControls() {
        etInputName = (EditText) findViewById(R.id.et_input_name);
        btNext = (Button) findViewById(R.id.bt_next);

        // 데이터 있으면 그 값으로 위젯 세팅
        if (!emptyDBTable()) {
            String name = getNameColumn();
            etInputName.setText(name);
        }

        btNext.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String name = String.valueOf(etInputName.getText());

                if (!emptyDBTable()) dbHelper.updateColumn(1, name);
                else dbHelper.insertColumn(name);

                Log.d(Const.TAG + activityName, "DB check :: name = " + getNameColumn());

                if (!checkNickName()) {
                    showToast(R.string.nickname_not_exist);
                    etInputName.setHintTextColor(getResources().getColor(R.color.red));
                } else {
                    startActivity(new Intent(getActivity(), MainActivity.class));
                }
            }
        });

    }

    private boolean checkNickName() {
        if (etInputName.getText().length() > 0) return true;
        else return false;
    }

}