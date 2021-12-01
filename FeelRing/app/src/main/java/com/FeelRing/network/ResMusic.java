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

import java.io.Serializable;

public class ResMusic implements Serializable {
    private String id;
    private String thumbnail;
    private String title;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getThumbnail() {
        return thumbnail;
    }

    public void setThumbnail(String thumbnail) {
        this.thumbnail = thumbnail;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

}
