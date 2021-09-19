package com.FeelRing.activity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.service.autofill.ImageTransformation;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.FeelRing.R;
import com.FeelRing.utils.Const;
import com.FeelRing.utils.ToastUtil;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends BaseActivity {
    private static final int PICK_FROM_CAMERA = 0;
    private static final int PICK_FROM_ALBUM = 1;
    private static final int CROP_FROM_iMAGE = 2;
    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 1001;

    // Camera
    String imageFilePath = "";
    Uri photoURI;
    Uri albumURI;
    File selectedImage;
    boolean isImageSelect = false;
    boolean isAlbum = false;

    // widget
    Button btTakePic;
    Button btReadPic;
    Button btOK;
    ImageView ivImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initControls();
    }

    private void initControls() {
        btTakePic = (Button) findViewById(R.id.bt_take_pic);
        btReadPic = (Button) findViewById(R.id.bt_read_pic);
        btOK = (Button) findViewById(R.id.bt_ok);
        ivImage = (ImageView) findViewById(R.id.iv_image) ;

        btTakePic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureCamera();
            }
        });

        btReadPic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getGalleryIntent();
            }
        });

    }

    private void captureCamera() {
        if (!checkPermission()) {
            showToast(R.string.check_permission);
        } else {
            String state = Environment.getExternalStorageState();
            if (Environment.MEDIA_MOUNTED.equals(state)) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                    File photoFile = null;
                    try {
                        photoFile = createImageFile();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    if (photoFile != null) {
                        photoURI = FileProvider.getUriForFile(this, "com.FeelRing.provider", photoFile);

                        Log.d("rsj", "photo file :: " + photoFile.toString());
                        Log.d("rsj", "photo uri :: " + photoURI.toString());

                        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                        startActivityForResult(takePictureIntent, PICK_FROM_CAMERA);
                    }
                }
            }
        }
    }

    private void getGalleryIntent() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        startActivityForResult(intent, PICK_FROM_ALBUM);
    }

    private void cropImage() {
        Intent cropIntent = new Intent("com.android.camera.action.CROP");
        cropIntent.setDataAndType(photoURI, "image/*");

        // CROP할 이미지를 200*200 크기로 저장
        cropIntent.putExtra("outputX", 200); // CROP한 이미지의 x축 크기
        cropIntent.putExtra("outputY", 200); // CROP한 이미지의 y축 크기
        cropIntent.putExtra("aspectX", 1); // CROP 박스의 X축 비율
        cropIntent.putExtra("aspectY", 1); // CROP 박스의 Y축 비율
        cropIntent.putExtra("scale", true);
        cropIntent.putExtra("output", albumURI);

        startActivityForResult(cropIntent, CROP_FROM_iMAGE);
    }
    
    // 사진 찍고 앨범에 저장
    private void galleryAddPic() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(imageFilePath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        this.sendBroadcast(mediaScanIntent);
    }

    private File createImageFile() throws IOException {
        //create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = timeStamp + ".jpg";
        File storageDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/feelRing/" + imageFileName);

        // save a file
        imageFilePath = storageDir.getAbsolutePath();
        Log.d("rsj", "image file path :: " + imageFilePath);
        return storageDir;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case PICK_FROM_CAMERA:
                isAlbum = false;
                cropImage();
                break;

            case PICK_FROM_ALBUM:
                isAlbum = true;
                File albumFile = null;
                try {
                    albumFile = createImageFile();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (albumFile != null) {
                    albumURI = Uri.fromFile(albumFile);
                }

                photoURI = data.getData();
                cropImage();
                break;

            case CROP_FROM_iMAGE:
                galleryAddPic();
                final Bundle extras = data.getExtras();
                showImageInView(extras, ivImage);
                break;
        }

    }

    private void showImageInView(Bundle extras, ImageView imageView) {
        if(extras != null) {

            Bitmap photo = extras.getParcelable("data"); // CROP된 BITMAP
            imageView.setImageBitmap(photo); // 레이아웃의 이미지칸에 CROP된 BITMAP을 보여줌
            //absoultePath = filePath;
        }

        // 임시 파일 삭제
        File f = new File(photoURI.getPath());
        if(f.exists()) f.delete();
    }

    private boolean checkPermission() {
        int permssionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA);

        if (permssionCheck == PackageManager.PERMISSION_GRANTED) return true;
        else return false;
    }

}