package com.FeelRing.network;

import java.io.File;

import okhttp3.Callback;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

public class NetworkManager {
    public static void requestTest(String url, Callback cb) {
        OkHttpClient client = new OkHttpClient();

        Request.Builder builder = new Request.Builder().url(url).get();
        Request request = builder.build();

        client.newCall(request).enqueue(cb);
    }

    public static void requestEmotion(String url, File image, Callback cb) {
        // 1. create request body
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", image.getName(), RequestBody.create(MultipartBody.FORM, image))
                .build();

        // 2. create request contained request body
        Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();

        // 3. send request
        OkHttpClient client = new OkHttpClient();
        client.newCall(request).enqueue(cb);
    }

}
