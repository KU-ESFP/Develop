package com.FeelRing.activity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

import com.FeelRing.R;
import com.FeelRing.network.NetworkManager;
import com.FeelRing.utils.Const;

import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class AnalysisActivity extends BaseActivity {
    final String activityName = "::AnalysisActivity";
    String photoPath;
    File photoFile;
    ArrayList<String> resFile;
    ArrayList<String> resMusic1;
    ArrayList<String> resMusic2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_analysis);

        Intent intent = getIntent();
        photoPath = intent.getStringExtra("photoPath");
        photoFile = new File(photoPath);

        if (photoFile != null) {
            Log.d(Const.TAG + activityName, "(3) take pic :: photo file is NOT null!! :: size = " + photoFile.length() / 1024 + "KB");
            //requestUploadFile("http://203.252.166.75:8080/api/test");
            //requestUploadFile(String.valueOf(R.string.upload_file_url));

            String emotion = "행복";
            Intent newIntent = new Intent(getActivity(), ResultActivity.class);
            newIntent.putExtra("emotion", emotion);
            startActivity(newIntent);
        } else {
            Log.d(Const.TAG + activityName, "(3) take pic :: photo file is null!!");
            finish();
        }

    }

    // 서버 통신 - 파일 업로드 요청
    private void requestUploadFile(String url) {
        NetworkManager.requestEmotion(url, photoFile, new Callback() {
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                Log.d(Const.TAG + activityName, "call fail(1)");
                showToast(R.string.fail_request);
                e.printStackTrace();
                finish();
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful())  {
                    Log.d(Const.TAG + activityName, "call success");
                    ResponseBody body = response.body();

                    // TODO(2): 결과 나오면 감정 다음 액티비티에 넘겨주기
                    if (body != null) {
                        String json = body.string();
                        Log.d(Const.TAG + activityName, "res json :: " + json);

                        try {
                            resFile  = new ArrayList<String>();

                            JSONObject jsonObject = new JSONObject(json);
                            resFile.add(jsonObject.getString("emotion"));
                            resFile.add(jsonObject.getString("fileName"));
                            resFile.add(jsonObject.getString("fileDownloadUri"));
                            resFile.add(jsonObject.getString("fileType"));
                            resFile.add(jsonObject.getString("size"));

                            Log.d(Const.TAG + activityName, "res json parse :: " + resFile.get(0) + " " + resFile.get(1) + " " + resFile.get(2) + " " + resFile.get(3) + " " + resFile.get(4));

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }

                    Intent intent = new Intent(getActivity(), ResultActivity.class);
                    intent.putExtra("fileInfo", resFile);
                    intent.putExtra("musicInfo1", resMusic1);
                    intent.putExtra("musicInfo2", resMusic1);
                    startActivity(intent);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {

                        }
                    });
                }
                else {
                    Log.d(Const.TAG + activityName, "call fail(2)");
                    showToast(R.string.fail_request);
                    finish();
                }
            }
        });
    }

    @Override
    public void onBackPressed() {
        showToast(R.string.no_back);
    }


}