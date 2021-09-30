package com.FeelRing.activity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
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
import com.FeelRing.network.NetworkManager;
import com.FeelRing.utils.Const;
import com.bumptech.glide.Glide;

import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;

public class MainActivity extends BaseActivity {
    Button bt_camera;
    Button bt_gallery;
    Button bt_ok;
    ImageView iv_photo;

    String mCurrentPhotoPath;
    Uri photoURI;
    Uri albumURI;
    File photoFile;

    boolean is_exist = false;

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
                captureCamera();
            }
        });

        bt_gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                readGallery();
            }
        });

        if (is_exist = true) bt_ok.setActivated(true);
        else bt_ok.setActivated(false);

        bt_ok.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                NetworkManager.requestTest("http://203.252.166.75:8080/api/test", new Callback() {
                    @Override
                    public void onFailure(@NotNull Call call, @NotNull IOException e) {
                        Log.d(Const.TAG, "call fail");
                    }

                    @Override
                    public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                        if (response.isSuccessful()) {
//                            ResponseBody body = response.body();
//                            if (body != null) Log.d(Const.TAG, "response == " + body.string());
                            String res = response.body().string();
                            Log.d(Const.TAG, "res == " + res);

                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {

                                }
                            });


                        } else {
                            Log.d(Const.TAG, "response error");
                        }
                    }
                });

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

    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);

        mCurrentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private void captureCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(takePictureIntent.resolveActivity(getPackageManager()) != null) {
            photoFile = null;

            try {
                photoFile = createImageFile();
                OutputStream out = new FileOutputStream(photoFile);

            } catch (IOException ex) {

            }

            if(photoFile != null) {
                photoURI = FileProvider.getUriForFile(this, "com.FeelRing.fileprovider", photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
            }
        }
    }

    private void readGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        startActivityForResult(intent, REQUEST_READ_PHOTO);
    }

    private void showImageView() {
        File file = new File(mCurrentPhotoPath);
        Bitmap bitmap;

        if (Build.VERSION.SDK_INT >= 29) {
            ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), Uri.fromFile(file));
            try {
                bitmap = ImageDecoder.decodeBitmap(source);
                if (bitmap != null) {
                    iv_photo.setImageBitmap(bitmap);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), Uri.fromFile(file));
                if (bitmap != null) {
                    iv_photo.setImageBitmap(bitmap);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        try {
            switch (requestCode) {
                case REQUEST_TAKE_PHOTO: {
                    showImageView();
                    is_exist = true;
                    break;
                }

                case REQUEST_READ_PHOTO: {
                    //showImageView();
                    Uri uri = data.getData();
                    Glide.with(getApplicationContext()).load(uri).into(iv_photo);
                    is_exist = true;

                    photoFile = createImageFile();
                    OutputStream out = new FileOutputStream(photoFile);

                    break;
                }
            }
        } catch (Exception error) {
            is_exist = false;
            error.printStackTrace();
        }

    }


}