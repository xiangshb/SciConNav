import pandas as pd
import numpy as np
import pymysql
from typing import Union, List
from tqdm import tqdm
import time
from sqlalchemy import create_engine
from config import DatabaseParams, calculate_runtime

class DatabaseManager(DatabaseParams):
        
    def __init__(self):
        super(DatabaseManager, self).__init__()
        self.db = None
        self.engine = None

    def connect(self, driver: str = 'pymysql') -> pymysql.connections.Connection:
        if driver == 'pymysql':
            self.db = pymysql.connect(host=self.server_ip, port = self.server_port, db=self.database_name, user=self.username, password=self.password, 
                                      charset=self.charset, cursorclass=pymysql.cursors.DictCursor)
            return self.db
        elif driver == 'sqlalchemy':
            self.engine = create_engine(f'mysql://{self.username}:{self.password}@{self.server_ip}:{self.server_port}/{self.database_name}?charset={self.charset}')
            return self.engine
        else: raise ValueError('Undefined driver')
        
    def close(self) -> None:
        if self.db is not None:
            self.db.close()
        if self.engine is not None:
            self.engine.dispose()
    
    @calculate_runtime
    def query_table(self, table_name: str, columns: Union[List[str], None] = None, 
                    join_tables: Union[List[str], None] = None, 
                    join_conditions: Union[List[str], None] = None, 
                    where_conditions: Union[List[str], None] = None,
                    batch_read: bool = False,
                    batch_size: int = 5000,
                    driver: str = 'sqlalchemy') -> pd.DataFrame:
        self.connect(driver) # 确保已连接到数据库
        column_clause = '*' if columns is None else ', '.join(columns) # 处理列名
        
        # 处理JOIN语句
        join_clause = ''
        if join_tables and join_conditions:
            assert len(join_tables) == len(join_conditions), "Join tables and conditions must match"
            for i, (table, condition) in enumerate(zip(join_tables, join_conditions)):
                join_clause += f"LEFT JOIN `{table}` ON {condition}"
                if i < len(join_tables) - 1:  
                    join_clause += ' '

        # 处理WHERE条件
        where_clause = ''
        if where_conditions:
            where_clause = 'WHERE ' + ' AND '.join(where_conditions)

        # 构建最终的SQL查询语句
        query = f"SELECT {column_clause} FROM `{table_name}` {join_clause} {where_clause}"

        # 执行SQL查询并获取DataFrame
        if driver == 'pymysql': # pymysql faster for batch_read
            with self.db.cursor() as cursor:
                cursor.execute(query)
                if batch_read:
                    all_batches = []
                    total_rows = self.get_total_rows(query)
                    batches = np.ceil(total_rows / batch_size).astype(int)
                    with tqdm(total=batches, desc='Reading batches', unit='batch') as pbar:
                        while True:
                            rows_batch = cursor.fetchmany(batch_size)
                            if not rows_batch: break
                            all_batches.extend(rows_batch)
                            pbar.update(1)
                    df = pd.DataFrame(all_batches, columns=[desc[0] for desc in cursor.description])
                else: 
                    result = cursor.fetchall()
                    df = pd.DataFrame(result, columns = [i[0] for i in cursor.description])
        elif driver == 'sqlalchemy': 
            if batch_read:
                df_batches = []
                with self.engine.connect() as connection:
                    total_rows = connection.execute(f"SELECT COUNT(*) FROM ( {query} ) AS subquery").scalar()
                batches = np.ceil(total_rows / batch_size).astype(int)
                with tqdm(total=batches, desc='Reading batches', unit='batch') as pbar:
                    for i in range(batches):
                        offset = i * batch_size
                        batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                        df_batch = pd.read_sql(batch_query, con=self.engine)
                        df_batches.append(df_batch)
                        pbar.update(1)
                df = pd.concat(df_batches, ignore_index=True)
            else: df = pd.read_sql(query, con=self.engine)
        else: raise ValueError('Undefined driver')
        self.close()
        return df

    def get_total_rows(self, query: str) -> int:
        count_query = f"SELECT COUNT(*) FROM ( {query} ) AS subquery"
        with self.db.cursor() as cursor:
            cursor.execute(count_query)
            count_result = cursor.fetchone()
            total_rows = count_result['COUNT(*)'] if count_result else 0
        return total_rows

# db_manager = DatabaseManager()

if __name__=='__main__':
    # db_manager = 
    table_results = DatabaseManager().query_table(driver= 'sqlalchemy', table_name = 'concepts', columns=['id', 'level', 'display_name'], batch_read=True) # driver= 'sqlalchemy' or pymysql
    print('Testing')

