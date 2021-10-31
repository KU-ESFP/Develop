package com.FeelRing.activity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.FeelRing.R;
import com.FeelRing.network.ResFile;
import com.FeelRing.network.ResMusic;
import com.FeelRing.utils.Const;

import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;

public class ResultActivity extends BaseActivity {
    final String activityName = "::ResultActivity";

    String name;

    // 인텐트 엑스트라 값
    String playlistId;
    ResFile fileInfo;
    ArrayList<ResMusic> musicInfo;

    // 감정 분석 결과 위젯 필드
    TextView tvName;
    TextView tvEmotion;

    // 노래 결과 위젯 필드
    LinearLayout llMusic1;
    LinearLayout llMusic2;
    ImageView ivThumnail1;
    ImageView ivThumnail2;
    TextView tvTitle1;
    TextView tvTitle2;

    // 버튼 위젯 필드
    Button btYoutube;
    Button btHome;

    String youtubeBasic;
    String youtubePlaylist;
    String playlistName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        initFields();
        initControls();
    }

    private void initFields() {
        name = getNameColumn();

        // 감정 분석 결과 위젯 필드
        tvName = (TextView) findViewById(R.id.tv_name);
        tvEmotion = (TextView) findViewById(R.id.tv_emotion);

        // 노래 결과 위젯 필드
        llMusic1 = (LinearLayout) findViewById(R.id.ll_music_1);
        llMusic2 = (LinearLayout) findViewById(R.id.ll_music_2);
        ivThumnail1 = (ImageView) findViewById(R.id.iv_thumnail_1);
        ivThumnail2 = (ImageView) findViewById(R.id.iv_thumnail_2);
        tvTitle1 = (TextView) findViewById(R.id.tv_title_1);
        tvTitle2 = (TextView) findViewById(R.id.tv_title_2);

        // 버튼 위젯 필드
        btYoutube = (Button) findViewById(R.id.bt_youtube);
        btHome = (Button) findViewById(R.id.bt_home);

        youtubeBasic = getResources().getString(R.string.youtube_basic);
        youtubePlaylist = getResources().getString(R.string.youtube_playlist);
        
        // 감정에 따라서 재생목록 이름 가져오기
//        playlistName = getResources().getString(R.string.pl_test);
//        // playListName = getPlayListName(emotion);

        // 인텐트 엑스트라 가져오기
        Intent intent = getIntent();
        musicInfo = new ArrayList<>();
        fileInfo = intent.getParcelableExtra("fileInfo");
        musicInfo = (ArrayList<ResMusic>) intent.getSerializableExtra("musicInfo");
        playlistId = intent.getStringExtra("playlistId");

        tvName.setText(name);
        tvEmotion.setText(fileInfo.getEmotion());

        tvTitle1.setText(musicInfo.get(0).getTitle());
        tvTitle1.setSelected(true);
        tvTitle2.setText(musicInfo.get(1).getTitle());
        tvTitle2.setSelected(true);

        new LoadImage().execute(musicInfo.get(0).getThumbnail(), musicInfo.get(1).getThumbnail());
    }

    private void initControls() {
        llMusic1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG + activityName, "click music 1");

                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(getMusicUri(musicInfo.get(0))));
                startActivity(intent);
            }
        });

        llMusic2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG + activityName, "click music 2");

                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(getMusicUri(musicInfo.get(1))));
                startActivity(intent);            }
        });

        btYoutube.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO: 감정에 따른 유튜브 재생목록 재생
                Log.d(Const.TAG + activityName, "click youtube");

                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(getPlaylistUrl()));
                startActivity(intent);
            }
        });

        btHome.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG + activityName, "click home");
                Intent intent = new Intent(getActivity(), MainActivity.class);
                startActivity(intent);
            }
        });
    }

    private String getMusicUri(ResMusic resMusic) {
        String youtubeUri = youtubeBasic + resMusic.getId();
        Log.d(Const.TAG + activityName, "check youtube url = " + youtubeUri);
        return youtubeUri;
    }

    private String getPlaylistUrl() {
        return youtubePlaylist + playlistId;
    }

    private String getPlaylistUrl2(String emotion) {
        String plName = "";

        switch (emotion) {
            case "happy":
                plName = getResources().getString(R.string.pl_happy);
                break;
            case "sad":
                plName = getResources().getString(R.string.pl_sad);
                break;
            case "surprise":
                plName = getResources().getString(R.string.pl_surprise);
                break;
            case "angry":
                plName = getResources().getString(R.string.pl_angry);
                break;
            case "neutral":
                plName = getResources().getString(R.string.pl_neutral);
                break;
        }

        return youtubePlaylist + plName;
    }

    public class LoadImage extends AsyncTask<String, String, Bitmap[]> {
        String className = "::LoadImage.class";
        Bitmap[] mBitmap = new Bitmap[2];

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            Log.d(Const.TAG + className, "preExcute ...");
        }

        @Override
        protected Bitmap[] doInBackground(String... args) {
            Log.d(Const.TAG + className, "download ...");

            Log.d(Const.TAG + className, "url 1 = " + args[0]);
            Log.d(Const.TAG + className, "url 2 = " + args[1]);

            try {
                mBitmap[0] = BitmapFactory.decodeStream((InputStream) new URL(args[0]).getContent());
                mBitmap[1] = BitmapFactory.decodeStream((InputStream) new URL(args[1]).getContent());
            } catch (Exception e) {
                e.printStackTrace();
            }
            return mBitmap;
        }

        @Override
        protected void onPostExecute(Bitmap[] image) {
            if (image != null) {
                ivThumnail1.setImageBitmap(image[0]);
                ivThumnail2.setImageBitmap(image[1]);
            } else {
                Log.d(Const.TAG + className, "image is null");
            }
        }

    }

}