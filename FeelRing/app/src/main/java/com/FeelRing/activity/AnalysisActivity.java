package com.FeelRing.activity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

import com.FeelRing.R;
import com.FeelRing.network.NetworkManager;
import com.FeelRing.network.ResFile;
import com.FeelRing.network.ResMusic;
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
    String requestUrl;
    ResFile resFile;
    ArrayList<ResMusic> resMusics;
    String playlistId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_analysis);

        Intent intent = getIntent();
        photoPath = intent.getStringExtra("photoPath");
        photoFile = new File(photoPath);
        requestUrl = getResources().getString(R.string.upload_file_url);


        if (photoFile != null) {
            Log.d(Const.TAG + activityName, "(3) take pic :: photo file is NOT null!! :: size = " + photoFile.length() / 1024 + "KB");
            requestUploadFile(requestUrl);

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
                        Log.d(Const.TAG, "res json:: " + json);
                        Log.d(Const.TAG, "res json length:: " + json.length());

                        try {
                            parsingJson(json);

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }

                    Intent intent = new Intent(getActivity(), ResultActivity.class);
                    intent.putExtra("playlistId", playlistId);
                    intent.putExtra("fileInfo", resFile);
                    intent.putExtra("musicInfo", resMusics);
                    startActivity(intent);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {

                        }
                    });
                }
                else {
                    Log.d(Const.TAG + activityName, "call fail(2)");
                    //showToast(R.string.fail_request);
                    finish();
                }
            }
        });
    }

    private void parsingJson(String json) throws JSONException {
        if (json.length() == 4) {
            // TODO: 이전 액티비티에 실패했다고 알려주기

            Log.d(Const.TAG, "json is null");
            finish();
        }

        JSONObject jsonObject = new JSONObject(json);

        // 플레이리스트 아이디
        playlistId = jsonObject.getString("playlist_id");

        // 파일 파싱
        JSONObject fileObject = jsonObject.getJSONObject("file");

        resFile = new ResFile();
        resFile.setEmotion(fileObject.getString("emotion"));
        resFile.setFileName(fileObject.getString("fileName"));
        resFile.setFileSize(fileObject.getString("fileSize"));
        resFile.setFileType(fileObject.getString("fileType"));

        // 음악 파싱
        resMusics = new ArrayList<>();
        for (int i = 1; i < 3; i++) {
            String music = "music" + i;
            JSONObject musicObject = jsonObject.getJSONObject(music);

            ResMusic resMusic = new ResMusic();
            resMusic.setId(musicObject.getString("id"));
            resMusic.setThumbnail(musicObject.getString("thumbnail"));
            resMusic.setTitle(musicObject.getString("title"));

            resMusics.add(resMusic);
        }

        Log.d(Const.TAG + activityName, playlistId + " " + resFile.getEmotion() + " " + resFile.getFileName() + " " + resFile.getFileSize() + " " + resFile.getFileType());

        for (int i = 0; i < resMusics.size(); i++) {
            Log.d(Const.TAG + activityName, resMusics.get(i).getId() + " " + resMusics.get(i).getThumbnail() + " " + resMusics.get(i).getTitle());
        }

    }

    @Override
    public void onBackPressed() {
        showToast(R.string.no_back);
    }


}