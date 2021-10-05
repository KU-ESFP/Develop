package com.FeelRing.activity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.FeelRing.R;
import com.FeelRing.datebase.DBOpenHelper;
import com.FeelRing.utils.Const;

import java.util.ArrayList;

public class ResultActivity extends BaseActivity {
    final String activityName = "::ResultActivity";

    String name;
    String emotion;
    DBOpenHelper dbHelper;

    // 인텐트 엑스트라 값
    ArrayList<String> fileInfo;
    ArrayList<String> musicInfo1;
    ArrayList<String> musicInfo2;

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
    String playListName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        Intent intent = getIntent();
        emotion = intent.getStringExtra("emotion");
        name = getNameColumn();

        // TODO: 실제론 어레이로 인텐트 받아옴 분석결과 / 음악1 / 음악2
        // fileInfo = intent.getStringArrayListExtra("fileInfo");
        // musicInfo1 = intent.getStringArrayListExtra("musicInfo1");
        // musicInfo2 = intent.getStringArrayListExtra("musicInfo2");

        initFields();
        initControls();
    }

    private void initFields() {
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
        
        // TODO: 감정에 따라서 재생목록 이름 가져오기
        playListName = getResources().getString(R.string.pl_test);
        // playListName = getPlayListName(emotion);

    }

    private void initControls() {
        tvName.setText(name);
        tvEmotion.setText(emotion);

        // TODO: 인텐트값 각 위젯에 설정
//        ivThumnail1.setImageBitmap(musicInfo1.썸네일1);
//        ivThumnail2.setImageBitmap(musicInfo2.썸네일2);
//        tvTitle1.setText(musicInfo1.타이틀1);
//        tvTitle2.setText(musicInfo1.타이틀2);

        llMusic1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO: 음악1 유튜브 링크 연결
                Log.d(Const.TAG + activityName, "click music 1");

                String youtubeUri = youtubeBasic + "4TWR90KJl84";
                Log.d(Const.TAG + activityName, "check youtube url = " + youtubeUri);

                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(youtubeUri));
                startActivity(intent);
            }
        });

        llMusic2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO: 음악2 유튜브 링크 연결
                Log.d(Const.TAG + activityName, "click music 2");

                String youtubeUri = youtubeBasic + "rrUxPFklKS8";
                Log.d(Const.TAG + activityName, "check youtube url = " + youtubeUri);

                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(youtubeUri));
                startActivity(intent);
            }
        });

        btYoutube.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO: 감정에 따른 유튜브 재생목록 재생
                Log.d(Const.TAG + activityName, "click youtube");

                String youtubeUri = youtubePlaylist + playListName;
                Log.d(Const.TAG + activityName, "check youtube url = " + youtubeUri);

                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(youtubeUri));
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

    private String getPlayListName(String emotion) {
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

        return plName;
    }
}