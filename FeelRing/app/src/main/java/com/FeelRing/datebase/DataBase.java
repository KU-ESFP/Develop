package com.FeelRing.datebase;

import android.provider.BaseColumns;

public final class DataBase {
    public static final class CreateDB implements BaseColumns {
        public static final String NAME = "name";
        public static final String _TABLENAME0 = "info";

        public static final String _CREATE0 = "create table if not exists "+_TABLENAME0+"("
                +_ID+" integer primary key autoincrement, "
                +NAME+" text not null);";
    }
}
