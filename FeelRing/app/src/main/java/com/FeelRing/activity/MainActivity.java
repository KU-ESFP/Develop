package com.FeelRing.activity;

import android.Manifest;
import android.app.AlertDialog;
import android.content.DialogInterface;
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
    final String activityName = "::MainActivity";

    // 위젯
    Button btPic;
    Button btOk;
    Button btEditName;
    ImageView ivPhoto;

    // 이미지 파일, 경로, uri
    File photoFile;
    String photoPath;
    Uri photoURI;


    final static int REQUEST_TAKE_PHOTO = 1;
    final static int REQUEST_READ_PHOTO = 2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initFields();
    }

    protected void onResume() {
        super.onResume();

        checkPermission();
        initControls();
    }

    private void initFields() {
        btPic = (Button) findViewById(R.id.bt_pic);
        ivPhoto = (ImageView) findViewById(R.id.iv_photo);
        btEditName = (Button) findViewById(R.id.bt_edit_name);
        btOk = (Button) findViewById(R.id.bt_ok);
    }

    private  void initControls() {

//        if (photoFile != null && photoURI != null) {
//            Log.d(Const.TAG, "photo File and URI is not null :: size =  " + photoFile.length() / 1024 + "kb :: uri = " + photoURI);
//            Glide.with(getApplicationContext()).load(photoURI).into(iv_photo);
//        }

        btPic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                createDialog();
            }
        });

        btEditName.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getActivity(), SurveyActivity.class));
            }
        });

        btOk.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(Const.TAG + activityName, "ok button click");

                if (photoFile != null) {
                    Log.d(Const.TAG + activityName, "file is not null :: request...");

                    Intent intent = new Intent(getActivity(), AnalysisActivity.class);
                    intent.putExtra("photoPath", photoPath);
                    startActivity(intent);

                    //requestUploadFile(String.valueOf(R.string.upload_file_url));
                }
            }
        });

    }

    private CharSequence[] imageChooser = {"사진 촬영", "갤러리에서 가져오기"};

    private void createDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("사진 선택");
        builder.setItems(imageChooser, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which) {
                    case 0:
                        captureCamera();
                        break;
                    case 1:
                        readGallery();
                        break;
                }
                dialog.dismiss();
            }
        });
        builder.create();
        builder.show();
    }

    // 권한 체크
    private void checkPermission() {
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                    && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                Log.d(Const.TAG + activityName, "권한 설정 완료");
            } else {
                Log.d(Const.TAG + activityName, "권한 설정 요청");
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},1);
            }
        }
    }

    // 권한 요청
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        Log.d(Const.TAG + activityName, "onRequestPermissionsResult");

        if (grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED ) {
            Log.d(Const.TAG + activityName, "Permission: " + permissions[0] + "was " + grantResults[0]);
        }
    }

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
                    Log.d(Const.TAG + activityName, "(1) capture photo uri == " + photoURI);
                    Glide.with(this).load(photoURI).into(ivPhoto);

                    if (photoFile != null) {
                        Log.d(Const.TAG + activityName, "(1) take pic :: photo file is NOT null!! :: size = " + photoFile.length() / 1024 + "KB");
                        btOk.setEnabled(true);
                    } else {
                        Log.d(Const.TAG, "(1) take pic :: photo file is null!! :: size = " + photoFile.length() / 1024 + "KB");
                        btOk.setEnabled(false);
                    }
                    break;
                }

                case REQUEST_READ_PHOTO: {
                    photoURI = data.getData();
                    setPhotoPath(photoURI);
                    Log.d(Const.TAG + activityName, "(2) gallery photo uri == " + photoURI);

                    Glide.with(getApplicationContext()).load(photoURI).into(ivPhoto);

                    if (photoFile != null) {
                        Log.d(Const.TAG + activityName, "(2) add pic :: photo file is NOT null!! :: size = " + photoFile.length() / 1024 + "KB");
                        btOk.setEnabled(true);
                    } else {
                        Log.d(Const.TAG + activityName, "(2) add pic :: photo file is null!! :: size = " + photoFile.length() / 1024 + "KB");
                        btOk.setEnabled(false);
                    }

                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}