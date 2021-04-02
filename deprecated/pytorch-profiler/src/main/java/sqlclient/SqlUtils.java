package sqlclient;

import java.util.Properties;

import com.alibaba.druid.pool.DruidDataSource;
import com.alibaba.druid.pool.DruidPooledConnection;

/**
 * @author wuheng@otcaix.iscas.ac.cn
 *
 * @version 1.2.0
 * @since   2020/4/23
 *
 */
public class SqlUtils {

    private SqlUtils() {
        super();
    }

    @SuppressWarnings("resource")
    public static DruidPooledConnection createConnection(String driver, String jdbc, String user, String pwd) throws Exception {
        Properties props = new Properties();
        props.put("druid.driverClassName", driver);
        props.put("druid.url", jdbc + "?autoReconnect=true&serverTimezone=GMT");
        props.put("druid.username", user);
        props.put("druid.password", pwd);
        props.put("druid.initialSize", 10);
        props.put("druid.maxActive", 100);
        props.put("druid.maxWait", 3000);
        props.put("druid.keepAlive", true);
        DruidDataSource dataSource = new DruidDataSource();
        dataSource.configFromPropety(props);
        return dataSource.getConnection();
    }

}
