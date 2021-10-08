package com.FeelRing.network;

/*
     {
      "file": {
        "emotion": "neutral",
        "fileName": "ryusuz.jpg",
        "fileSize": 92292,
        "fileType": ".jpg"
      },
      "music1": {
        "id": "_X3r09dgbQg",
        "thumbnail": "https://i.ytimg.com/vi/_X3r09dgbQg/hq720.jpg?sqp=-oaymwEcCOgCEMoBSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLCfpF4rgM6dyYDulUj4L4Qsj7GcIA",
        "title": "ASAP-STAYC(\uc2a4\ud14c\uc774\uc528)"
      },
      "music2": {
        "id": "D1PvIWdJ8xo",
        "thumbnail": "https://i.ytimg.com/vi/D1PvIWdJ8xo/hq720.jpg?sqp=-oaymwEcCOgCEMoBSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLATIkAFT4yGDy_-LpA3bKQkIcm6Aw",
        "title": "Blueming-\uc544\uc774\uc720"
      }
    }
 */

import android.os.Parcel;
import android.os.Parcelable;

public class ResFile implements Parcelable {
    private String emotion;
    private String fileName;
    private String fileSize;
    private String fileType;

    public String getEmotion() {
        return emotion;
    }

    public void setEmotion(String emotion) {
        this.emotion = emotion;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public String getFileSize() {
        return fileSize;
    }

    public void setFileSize(String fileSize) {
        this.fileSize = fileSize;
    }

    public String getFileType() {
        return fileType;
    }

    public void setFileType(String fileType) {
        this.fileType = fileType;
    }


    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(this.emotion);
        dest.writeString(this.fileName);
        dest.writeString(this.fileSize);
        dest.writeString(this.fileType);
    }

    public ResFile() {
    }

    protected ResFile(Parcel in) {
        this.emotion = in.readString();
        this.fileName = in.readString();
        this.fileSize = in.readString();
        this.fileType = in.readString();
    }

    public static final Parcelable.Creator<ResFile> CREATOR = new Parcelable.Creator<ResFile>() {
        @Override
        public ResFile createFromParcel(Parcel source) {
            return new ResFile(source);
        }

        @Override
        public ResFile[] newArray(int size) {
            return new ResFile[size];
        }
    };
}
