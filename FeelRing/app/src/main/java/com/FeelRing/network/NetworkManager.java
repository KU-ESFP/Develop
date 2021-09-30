package com.FeelRing.network;

import android.util.Log;

import com.FeelRing.utils.Const;

import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class NetworkManager {
    public static void requestTest(String url, Callback cb) {
        OkHttpClient client = new OkHttpClient();

        Request.Builder builder = new Request.Builder().url(url).get();
        Request request = builder.build();

        client.newCall(request).enqueue(new Callback() {
            //Handler mainHandler = new Handler();

            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                Log.d(Const.TAG, "call fail");
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    ResponseBody body = response.body();
                    if (body != null) Log.d(Const.TAG, "response == " + body.string());
                } else {
                    Log.d(Const.TAG, "response error");
                }
            }
        });
    }

    public static void post(String requestURL, File image, ResEmotion resEmotion) {
        // 1. create request body
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", image.getName(), RequestBody.create(MultipartBody.FORM, image))
                .build();

        // 2. create request contained request body
        Request request = new Request.Builder()
                .url(requestURL)
                .post(requestBody)
                .build();

        // 3. send request
        OkHttpClient client = new OkHttpClient();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                Log.d(Const.TAG, "check response body :: " + response.body().string());

                try {
                    // 4. parse response
                    JSONObject body = new JSONObject(response.body().string());
                    String emotion  = body.getString("emotion");
                    resEmotion.setEmotion(emotion);
                    Log.d(Const.TAG, "check emotion :: " + resEmotion.getEmotion());

                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        });

    }

}
