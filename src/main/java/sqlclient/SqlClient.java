package sqlclient;


import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.logging.Logger;

import com.alibaba.druid.pool.DruidPooledConnection;

/**
 * @author wuheng@otcaix.iscas.ac.cn
 *
 * @version 1.2.0
 * @since   2020/4/23
 *
 */
public class SqlClient {

    public static final Logger m_logger = Logger.getLogger(SqlClient.class.getName());

    public static final String DEFAULT_DB       = "kube";

    public static final String LABEL_DATABASE   = "#DATBASE#";

    public static final String LABEL_TABLE      = "#TABLE#";

    public static final String LABEL_NAME       = "#NAME#";

    public static final String LABEL_NAMESPACE  = "#NAMESPACE#";

    public static final String LABEL_JSON       = "#JSON#";




    public static final String CHECK_DATABASE  = "SELECT * FROM information_schema.SCHEMATA where SCHEMA_NAME='#DATBASE#'";

    public static final String CREATE_DATABASE = "CREATE DATABASE #DATBASE#";

    public static final String DELETE_DATABASE = "DROP DATABASE #DATBASE#";


    public static final String CHECK_TABLE     = "SELECT DISTINCT t.table_name, n.SCHEMA_NAME FROM "
            + "information_schema.TABLES t, information_schema.SCHEMATA n "
            + "WHERE t.table_name = '#TABLE#' AND n.SCHEMA_NAME = '#DATBASE#'";

    public static final String CREATE_TABLE    = "CREATE TABLE #TABLE# (name varchar(250), namespace varchar(250), data json, primary key(name, namespace))";

    public static final String DELETE_TABLE    = "DROP TABLE #TABLE#";






    public static final String INSERT_OBJECT   = "INSERT INTO #TABLE# VALUES ('#NAME#', '#NAMESPACE#', '#JSON#')";

    public static final String UPDATE_OBJECT   = "UPDATE #TABLE# SET data = '#JSON#' WHERE name = '#NAME#' and namespace = '#NAMESPACE#'";

    public static final String DELETE_OBJECT   = "DELETE FROM #TABLE# WHERE name = '#NAME#' and namespace = '#NAMESPACE#'";

    /****************************************************************************
     *
     *
     *                         Basic
     *
     *
     *****************************************************************************/

    /**
     * conn
     */
    protected final DruidPooledConnection conn;

    protected final String database;

    public SqlClient(DruidPooledConnection conn) {
        this(conn, DEFAULT_DB);
    }


    public SqlClient(DruidPooledConnection conn, String database) {
        super();
        this.conn = conn;
        this.database = database;
    }



    /**
     * @return                conn
     */
    public DruidPooledConnection getConn() {
        return conn;
    }


    /****************************************************************************
     *
     *
     *                         Database, Table
     *
     *
     *****************************************************************************/
    /**
     * @param name                db name
     * @return                    true or false
     * @throws Exception          exception
     */
    @Deprecated
    public synchronized boolean hasDatabase(String name) throws Exception {
        return execWithResultCheck(null, CHECK_DATABASE.replace(LABEL_DATABASE, name));
    }

    /**
     * create database
     *
     * @throws Exception mysql exception
     */
    @Deprecated
    public synchronized boolean createDatabase(String name) throws Exception {
        return exec(null, CREATE_DATABASE.replace(LABEL_DATABASE, name));
    }

    /**
     * @return delete database
     * @throws Exception mysql exception
     */
    @Deprecated
    public synchronized boolean dropDatabase(String name) throws Exception {
        return exec(null, DELETE_DATABASE.replace(LABEL_DATABASE, name));
    }

    /**
     * @param name  class name
     * @return true if the table exists, otherwise return false
     * @throws Exception mysql exception
     */
    @Deprecated
    public synchronized boolean hasTable(String dbName, String tableName) throws Exception {
        return execWithResultCheck(dbName, CHECK_TABLE.replace(LABEL_DATABASE, dbName)
                .replace(LABEL_TABLE, tableName));
    }

    /**
     * @param clazz class
     * @return sql
     * @throws Exception mysql exception
     */
    @Deprecated
    public synchronized boolean createTable(String dbName, String tableName) throws Exception {
        return exec(dbName, CREATE_TABLE.replace(LABEL_TABLE, tableName));
    }

    /**
     * @param clazz class
     * @return sql
     * @throws Exception mysql exception
     */
    @Deprecated
    public synchronized boolean dropTable(String dbName, String tableName) throws Exception {
        return exec(dbName, DELETE_TABLE.replace(LABEL_TABLE, tableName));
    }


    /**
     * @param name                db name
     * @return                    true or false
     * @throws Exception          exception
     */
    public synchronized boolean hasDatabase() throws Exception {
        return execWithResultCheck(null, CHECK_DATABASE.replace(LABEL_DATABASE, database));
    }

    /**
     * create database
     *
     * @throws Exception mysql exception
     */
    public synchronized boolean createDatabase() throws Exception {
        return exec(null, CREATE_DATABASE.replace(LABEL_DATABASE, database));
    }

    /**
     * @return delete database
     * @throws Exception mysql exception
     */
    public synchronized boolean dropDatabase() throws Exception {
        return exec(null, DELETE_DATABASE.replace(LABEL_DATABASE, database));
    }

    /**
     * @param name  class name
     * @return true if the table exists, otherwise return false
     * @throws Exception mysql exception
     */
    public synchronized boolean hasTable(String tableName) throws Exception {
        return execWithResultCheck(database, CHECK_TABLE.replace(LABEL_DATABASE, database)
                .replace(LABEL_TABLE, tableName));
    }

    /**
     * @param clazz class
     * @return sql
     * @throws Exception mysql exception
     */
    public synchronized boolean createTable(String tableName) throws Exception {
        return exec(database, CREATE_TABLE.replace(LABEL_TABLE, tableName));
    }

    /**
     * @param clazz class
     * @return sql
     * @throws Exception mysql exception
     */
    public synchronized boolean dropTable(String tableName) throws Exception {
        return exec(database, DELETE_TABLE.replace(LABEL_TABLE, tableName));
    }



    /****************************************************************************
     *
     *
     *                         Insert, Update, Delete objects
     *
     *
     *****************************************************************************/
    /**
     * @param table                                  table
     * @param name                                   name
     * @param namespace                              namespace
     * @param json                                   json
     * @return                                       true or false
     * @throws Exception                             exception
     */
    public boolean insertObject(String table, String name, String namespace, String json) throws Exception {
        if(!exec(database, INSERT_OBJECT
                .replace(SqlClient.LABEL_TABLE, table)
                .replace(SqlClient.LABEL_NAME, name)
                .replace(SqlClient.LABEL_NAMESPACE, namespace)
                .replace(SqlClient.LABEL_JSON, json))) {
            return updateObject(table, name, namespace, json);
        }
        return true;
    }

    /**
     * @param table                                  table
     * @param name                                   name
     * @param namespace                              namespace
     * @param json                                   json
     * @return                                       true or false
     * @throws Exception                             exception
     */
    public boolean updateObject(String table, String name, String namespace, String json) throws Exception {
        return exec(database, UPDATE_OBJECT
                .replace(SqlClient.LABEL_TABLE, table)
                .replace(SqlClient.LABEL_NAME, name)
                .replace(SqlClient.LABEL_NAMESPACE, namespace)
                .replace(SqlClient.LABEL_JSON, json));
    }

    /**
     * @param table                                  table
     * @param name                                   name
     * @param namespace                              namespace
     * @param json                                   json
     * @return                                       true or false
     * @throws Exception                             exception
     */
    public boolean deleteObject(String table, String name, String namespace, String json) throws Exception {
        return exec(database, DELETE_OBJECT
                .replace(SqlClient.LABEL_TABLE, table)
                .replace(SqlClient.LABEL_NAME, name)
                .replace(SqlClient.LABEL_NAMESPACE, namespace)
                .replace(SqlClient.LABEL_JSON, json));
    }


    /****************************************************************************
     *
     *
     *                         Common
     *
     *
     *****************************************************************************/

    /**
     * @param dbName                          dbName
     * @param sql                             sql
     * @return                                true or false
     * @throws Exception                      exception
     */
    public boolean exec(String dbName, String sql) throws Exception {

        if (dbName != null) {
            conn.setCatalog(dbName);
        }

        PreparedStatement pstmt = null;


        try {
            pstmt = conn.prepareStatement(sql);
            return pstmt.execute();
        } catch (Exception ex) {
            m_logger.severe("caused by " + sql + ":" + ex);
            return false;
        } finally {
            if (pstmt != null) {
                pstmt.close();
            }
        }
    }
    /**
     * @param dbName                          dbName
     * @param sql                             sql
     * @return                                true or false
     * @throws Exception                      exception
     */
    public boolean execWithResultCheck(String dbName, String sql) throws Exception {
        if (dbName != null) {
            conn.setCatalog(dbName);
        }

        PreparedStatement pstmt = null;
        try {
            pstmt = conn.prepareStatement(sql);
            ResultSet rs = pstmt.executeQuery();
            return rs.next();
        } catch (Exception ex) {
            return false;
        } finally {
            if (pstmt != null) {
                pstmt.close();
            }
        }
    }

    /**
     * @param dbName                          dbName
     * @param sql                             sql
     * @return                                true or false
     * @throws Exception                      exception
     */
    public ResultSet execWithResult(String dbName, String sql) throws Exception {
        if (dbName != null) {
            conn.setCatalog(dbName);
        }

        try {
            return conn.prepareStatement(sql).executeQuery();
        } catch (Exception ex) {
            return null;
        }
    }

}