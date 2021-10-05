package com.FeelRing.datebase;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteDatabase.CursorFactory;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import com.FeelRing.utils.Const;

public class DBOpenHelper {
    private static final String DATABASE_NAME = "FeelRingDB.db";
    private static final int DATABASE_VERSION = 1;
    public static SQLiteDatabase mDB;
    private DatabaseHelper mDBHelper;
    private Context mCtx;

    // DDL
    private class DatabaseHelper extends SQLiteOpenHelper {

        public DatabaseHelper(Context context, String name, CursorFactory factory, int version) {
            super(context, name, factory, version);
        }

        @Override
        public void onCreate(SQLiteDatabase db){
            db.execSQL(DataBase.CreateDB._CREATE0);
        }

        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion){
            db.execSQL("DROP TABLE IF EXISTS "+DataBase.CreateDB._TABLENAME0);
            onCreate(db);
        }
    }

    public DBOpenHelper(Context context){
        this.mCtx = context;
    }

    public DBOpenHelper open() throws SQLException {
        mDBHelper = new DatabaseHelper(mCtx, DATABASE_NAME, null, DATABASE_VERSION);
        mDB = mDBHelper.getWritableDatabase();
        return this;
    }

    public void create(){
        mDBHelper.onCreate(mDB);
    }

    public void close(){
        mDB.close();
    }

    // DML
    public long insertColumn(String name){
        Log.d(Const.TAG, "DB :: insert name = " + name);

        ContentValues values = new ContentValues();
        values.put(DataBase.CreateDB.NAME, name);
        return mDB.insert(DataBase.CreateDB._TABLENAME0, null, values);
    }

    public Cursor selectColumns(){
        Log.d(Const.TAG, "DB :: select");

        return mDB.query(DataBase.CreateDB._TABLENAME0, null, null, null, null, null, null);
    }

    public boolean updateColumn(long id, String name){
        Log.d(Const.TAG, "DB :: update name = " + name);

        ContentValues values = new ContentValues();
        values.put(DataBase.CreateDB.NAME, name);
        return mDB.update(DataBase.CreateDB._TABLENAME0, values, "_id=" + id, null) > 0;
    }

    public void deleteAllColumns() {
        Log.d(Const.TAG, "DB :: delete all");

        mDB.delete(DataBase.CreateDB._TABLENAME0, null, null);
    }

    public boolean deleteOneColumn(long id){
        Log.d(Const.TAG, "DB :: delete one id = " + id);

        return mDB.delete(DataBase.CreateDB._TABLENAME0, "_id="+id, null) > 0;
    }

    // order by
    public Cursor sortColumn(String sort){
        Cursor c = mDB.rawQuery( "SELECT * FROM " + DataBase.CreateDB._TABLENAME0 + " ORDER BY " + sort + ";", null);
        return c;
    }

    public int getCountRecord() {
        int count = 0;
        Cursor cursor = mDB.rawQuery("SELECT * FROM " + DataBase.CreateDB._TABLENAME0, null);
        count = cursor.getCount();
        return count;
    }


}
