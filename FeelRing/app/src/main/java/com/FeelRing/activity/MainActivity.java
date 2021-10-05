package com.FeelRing.activity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import com.FeelRing.R;
import com.FeelRing.utils.Const;
import com.bumptech.glide.Glide;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends BaseActivity {
    // 위젯
    Button bt_camera;
    Button bt_gallery;
    Button bt_ok;
    ImageView iv_photo;

    // 이미지 파일, 경로, uri
    File photoFile;
    String photoPath;
    Uri photoURI;

    //ArrayList<String> resFile;

    final static int REQUEST_TAKE_PHOTO = 1;
    final static int REQUEST_READ_PHOTO = 2;
    final static int REQUEST_CROP_PHOTO = 3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initControls();
    }

    protected void onResume() {
        super.onResume();

        checkPermission();
    }

    private  void initControls() {
        bt_camera = (Button) findViewById(R.id.bt_camera);
        bt_gallery = (Button) findViewById(R.id.bt_gallery);
        bt_ok = (Button) findViewById(R.id.bt_ok);

        iv_photo = (ImageView) findViewById(R.id.iv_photo);

        checkPermission();

        bt_camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG, "camera button click");
                captureCamera();
            }
        });

        bt_gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG, "gallery button click");
                readGallery();
            }
        });

        bt_ok.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG, "ok button click");

                if (photoFile != null) {
                    Log.d(Const.TAG, "file is not null :: request...");

                    Intent intent = new Intent(getActivity(), AnalysisActivity.class);
                    intent.putExtra("photoPath", photoPath);
                    startActivity(intent);

                    //requestUploadFile(String.valueOf(R.string.upload_file_url));
                }
            }
        });
    }

    // 권한 체크
    private void checkPermission() {
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                    && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                Log.d(Const.TAG, "권한 설정 완료");
            } else {
                Log.d(Const.TAG, "권한 설정 요청");
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},1);
            }
        }
    }

    // 권한 요청
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        Log.d(Const.TAG, "onRequestPermissionsResult");

        if (grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED ) {
            Log.d(Const.TAG, "Permission: " + permissions[0] + "was " + grantResults[0]);
        }
    }

    // 서버 통신 - 파일 업로드 요청
//    private void requestUploadFile(String url) {
//        NetworkManager.requestEmotion(url, photoFile, new Callback() {
//            @Override
//            public void onFailure(@NotNull Call call, @NotNull IOException e) {
//                Log.d(Const.TAG, "call fail(1)");
//                e.printStackTrace();
//            }
//
//            @Override
//            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
//                if (response.isSuccessful())  {
//                    Log.d(Const.TAG, "call success");
//                    ResponseBody body = response.body();
//
//                    // TODO(1): 분석하기 누르면 응답 기다리는 동안 프로그레스바 실행 및 화면 못 만지게 멈추기
//                    // TODO(2): 결과 나오면 감정 다음 액티비티에 넘겨주기
//                    if (body != null) {
//                        String json = body.string();
//                        Log.d(Const.TAG, "res json :: " + json);
//
//                        try {
//                            resFile  = new ArrayList<String>();
//
//                            JSONObject jsonObject = new JSONObject(json);
//                            resFile.add(jsonObject.getString("emotion"));
//                            resFile.add(jsonObject.getString("fileName"));
//                            resFile.add(jsonObject.getString("fileDownloadUri"));
//                            resFile.add(jsonObject.getString("fileType"));
//                            resFile.add(jsonObject.getString("size"));
//
//                            Log.d(Const.TAG, "res json parse :: " + resFile.get(0) + " " + resFile.get(1) + " " + resFile.get(2) + " " + resFile.get(3) + " " + resFile.get(4));
//
//                        } catch (JSONException e) {
//                            e.printStackTrace();
//                        }
//                    }
//
//                    runOnUiThread(new Runnable() {
//                        @Override
//                        public void run() {
//
//                        }
//                    });
//                }
//                else Log.d(Const.TAG, "call fail(2)");
//            }
//        });
//    }

    // 카메라로 찍은 사진 저장하기 위한 파일 생성
    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);

        photoPath = image.getAbsolutePath();
        return image;
    }

    // 카메라 실행 및 사진 저장
    private void captureCamera() {
        //TODO(1): 다시 찍기 눌렀을 때 오류 해결하기
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(takePictureIntent.resolveActivity(getPackageManager()) != null) {
            photoFile = null;

            try {
                photoFile = createImageFile();
                OutputStream out = new FileOutputStream(photoFile);

            } catch (IOException e) {
                e.printStackTrace();
            }

            if(photoFile != null) {
                photoURI = FileProvider.getUriForFile(this, "com.FeelRing.fileprovider", photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
            }
        }
    }

    // 갤러리 열고 사진 가져오기
    private void readGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        startActivityForResult(intent, REQUEST_READ_PHOTO);
    }
    
    // uri로부터 파일 절대경로 알아오는 메서드
    private String getRealPathFromURI(Uri contentUri) {
        String[] proj = { MediaStore.Images.Media.DATA };
        Cursor cursor = getContentResolver().query(contentUri, proj, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(column_index);
    }
    
    // 절대경로 설정해서 포토파일에 덮어씌우는 메서드
    private void setPhotoPath(Uri photoURI) {
        photoPath = getRealPathFromURI(photoURI);
        photoFile = new File(photoPath);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        try {
            switch (requestCode) {
                case REQUEST_TAKE_PHOTO: {
                    Log.d(Const.TAG, "(1) capture photo uri == " + photoURI);
                    Glide.with(this).load(photoURI).into(iv_photo);

                    if (photoFile != null) {
                        Log.d(Const.TAG, "(1) take pic :: photo file is NOT null!! :: size = " + photoFile.length() / 1024 + "KB");
                        bt_ok.setEnabled(true);
                    } else {
                        Log.d(Const.TAG, "(1) take pic :: photo file is null!! :: size = " + photoFile.length() / 1024 + "KB");
                        bt_ok.setEnabled(false);
                    }
                    break;
                }

                case REQUEST_READ_PHOTO: {
                    photoURI = data.getData();
                    setPhotoPath(photoURI);

                    Log.d(Const.TAG, "(2) gallery photo uri == " + photoURI);
                    Glide.with(getApplicationContext()).load(photoURI).into(iv_photo);

                    if (photoFile != null) {
                        Log.d(Const.TAG, "(2) add pic :: photo file is NOT null!! :: size = " + photoFile.length() / 1024 + "KB");
                        bt_ok.setEnabled(true);
                    } else {
                        Log.d(Const.TAG, "(2) add pic :: photo file is null!! :: size = " + photoFile.length() / 1024 + "KB");
                        bt_ok.setEnabled(false);
                    }

                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


}