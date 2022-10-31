'''
Created on Fri Dec 10 10:35
@author: han
'''

import pymysql
import pandas as pd
from logs import paiplog

class PyDBconnector():
    '''
    for db connect : singleton design
    @params
    host, user, passwd, port, db, charset = 'utf8'
    '''
    _instance = None
    _conn = None
    _cur = None

    @classmethod
    def _getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls._getInstance
        return cls._instance

    def connect(self, host : str , user : str , passwd : str , port : int , db : str , charset : str ):
        if self._conn is None :
            try :
                self._conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, port=port, charset=charset)
                self._cur = self._conn.cursor()
            except :
                raise Exception('db connection error')

    def __init__(self, host : str = '127.0.0.1', user : str = 'paip', passwd : str = 'paip!123', port : int = 3306, db : str = 'paip', charset : str ='utf8'):
        self.connect(host, user , passwd , port , db ,charset)

    @paiplog
    def select_from_db(self, query : str) -> pd.DataFrame:
        if not query.lower().startswith('select'):
            raise Exception(f'query must be started with select : {query}')

        try:
            self._cur.execute(query)
            num_fields = len(self._cur.description)
            field_names = [i[0] for i in self._cur.description]
            return pd.DataFrame(self._cur.fetchall(), columns=field_names)
        except:
            raise Exception(f'select query execute error : {query}')

    @paiplog
    def insert_to_db(self, query : str):
        if not query.lower().startswith('insert'):
            raise Exception('insert query must be started with insert')

        # insert table 제한
        # if query.lower().find('tbl_pixel_stats') < 0 :
        #     raise Exception('insert table must be tbl_weight_stats')

        # insert
        try:

            resultNum = self._cur.execute(query)
            if resultNum > 0 : self._conn.commit()
            return resultNum
        except:
            raise Exception(f'insert query execute error : {query}')

    def get_cursor(self):
        return self._cur

    def close(self):
        self._conn.close()
        self._cur = None

if __name__ == '__main__':
    from datetime import datetime
    ## db test - gateway DB default
    from datetime import date, timedelta
    house_id, module_id = 'H01', 'CT01,6'
    sql_str = f'''select create_time, max(medianWeight) medianWeight,max(deltaWeight) deltaWeight,max(medianPixel) medianPixel
                    from tbl_weight_stats 
                    where 1=1
                    group by create_time
                    order by create_time desc
                    limit 1
                    '''


    dbConn = PyDBconnector()
    # dbConn = PyDBconnector()
    rows = dbConn.select_from_db(sql_str)
    print(rows)
    print(int((date.today() - datetime.strptime(rows.values[0][0], "%Y-%m-%d").date()).days))
    print(rows.columns)

    from sklearn.linear_model import LinearRegression
    X, y = rows[['medianPixel']], rows['medianWeight']
    line_fitter = LinearRegression()
    line_fitter.fit(X, y)

    print(f"prediction : {line_fitter.predict(pd.DataFrame([4000]))}")
    # #insert_str = f"insert into tbl_weight_stats values('2021-12-28','H01','CT01,6','404.9', '30', '5303')"
    # #returnVal = dbConn.insert_to_db(insert_str)
    #
    # joined_df = pd.read_csv('../util/joined_gp.csv')
    # print(joined_df)
    # print('insert results===========================')
    #
    # insert_string = f"insert into tbl_weight_stats values('2021-12-28','H01','CT01,6','404.9', '30', '5303')"
    # for pdRow in joined_df.iterrows() :
    #
    #     insert_string = f"insert into tbl_weight_stats(create_time, house_id, module_id, medianWeight, deltaWeight, medianPixel) values('{pdRow[1]['create_time']}','{pdRow[1]['house_id']}','{pdRow[1]['module_id']}',{pdRow[1]['weightMean']},{pdRow[1]['weightMeanStd']},{pdRow[1]['pixelMean']})"
    #     #print(insert_string)
    #     dbConn.insert_to_db(insert_string)

    dbConn.close()




