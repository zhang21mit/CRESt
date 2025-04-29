import logging
import os
from contextlib import contextmanager

import pandas as pd
import sigfig
from sqlalchemy import create_engine, MetaData, Table, Column, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database

from utils.utils import log_and_print, config_loader, get_logger


class Database:
    def __init__(self, exp_name, to_log=True, user='li', password='nano'):
        """
        :param exp_name: experiment name
        :param to_log: whether to log the database operations
        """

        self.configs = config_loader(exp_name, config_type='database_configs')
        self.tables_dict = self.configs['tables_dict']
        self.schema_list = self.configs['schema_list']
        self.metadata = MetaData()
        self.exp_name = exp_name
        self.to_log = to_log

        # logger settings
        self.logger = get_logger(exp_name=exp_name, module_name=self.__class__.__name__) if self.to_log else None

        # initialize database
        self.engine = create_engine(f'postgresql://{user}:{password}@10.140.0.20:5432/{exp_name}')
        self.Base = declarative_base(bind=self.engine, metadata=MetaData())

        # Define table classes
        self.table_classes = {}
        for table_full_name, table_info in self.tables_dict.items():
            schema = table_info['schema']
            table_name = table_info['table_name']
            columns = [Column(col_name, **col_info) for col_name, col_info in table_info['columns_dict'].items()]
            table_class = type(table_name, (self.Base,), {
                '__tablename__': table_name,
                '__table__': Table(table_name, self.metadata, *columns, schema=schema),
                '__init__': lambda table, **kwargs: self.setattr_from_dict(table, kwargs)
            })
            # note that the element in this dict is DeclarativeMeta object
            # while table_class.__table__ is the actual Table object
            self.table_classes[table_name] = table_class

        self.initialize_db()
        self.dbConnection = self.engine.connect()

    @contextmanager
    def session_scope(self):
        _engine = create_engine(f'postgresql://li:nano@10.140.0.20:5432/{self.exp_name}')
        _session = sessionmaker(bind=_engine)()
        yield _session
        try:
            _session.commit()
        except Exception as e:
            _session.rollback()
            print(f'session rolled back because of error: \n{e}')
        finally:
            _session.close()

    @staticmethod
    def setattr_from_dict(obj, attr_dict):
        for key, value in attr_dict.items():
            setattr(obj, key, value)

    def check_table_exist_on_server(self, table_dict):
        return self.get_table_on_server(table_dict) is not None

    def check_view_exist_on_server(self, view_name, schema='active_learning'):
        inspector = inspect(self.engine)
        views = inspector.get_view_names(schema=schema)
        return view_name in views

    def get_table_on_server(self, table_dict):
        metadata = MetaData(bind=self.engine, schema=table_dict['schema'])
        metadata.reflect()
        return metadata.tables.get(f'{table_dict["schema"]}.{table_dict["table_name"]}')

    def create_table(self, table_dict):
        if not self.check_table_exist_on_server(table_dict):
            self.table_classes[table_dict['table_name']].__table__.create(bind=self.engine)
        else:
            raise Exception(f'Table {table_dict["table_name"]} already exists in schema {table_dict["schema"]}!')

    def create_all_tables(self):
        self.metadata.create_all(bind=self.engine)

    def initialize_db(self):
        # create the database if the database not exists yet
        if self.check_db_exist():
            log_and_print(self.to_log, self.logger, 'info', f"Database {self.exp_name} connected successfully!")
        else:
            msg = f"Database {self.exp_name} does not exist yet, creating now..."
            log_and_print(self.to_log, self.logger, 'warning', msg)
            create_database(self.engine.url, template='template0')
            log_and_print(self.to_log, self.logger, 'info', f"Database {self.exp_name} connected successfully!")
        # create schemas if they do not exist yet
        for schema in self.schema_list:
            self.engine.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
            log_and_print(self.to_log, self.logger, 'info', f"Schema {schema} created successfully!")

        # create tables if they do not exist yet
        for table_full_name, table_dict in self.tables_dict.items():
            if not self.check_table_exist_on_server(table_dict):
                msg = f"Table {table_dict['table_name']} does not exist on server yet, creating now..."
                log_and_print(self.to_log, self.logger, 'warning', msg)
                self.create_table(table_dict)
                msg = f"Table {table_dict['table_name']} created on server successfully!"
                log_and_print(self.to_log, self.logger, 'info', msg)
            # check column consistency if the table already exists
            else:
                msg = f"Table {table_dict['table_name']} already exists on server, checking column consistency..."
                log_and_print(self.to_log, self.logger, 'info', msg)
                self.check_table_columns_match_btw_local_and_server(table_dict)
                msg = f"Table {table_dict['table_name']} column consistency checked successfully!"
                log_and_print(self.to_log, self.logger, 'info', msg)

    def check_db_exist(self):
        try:
            self.engine.connect()
            self.engine.dispose()
            return True
        except OperationalError as e:
            if 'does not exist' in str(e):
                return False

    def check_table_columns_match_btw_local_and_server(self, table_dict):
        """
        only columns name are checked, since type could be different between local and server,
        e.g. local Datetime vs server Timestamp, although they are both datetime type
        """
        table_name = table_dict['table_name']
        table_class = self.table_classes[table_name]
        local_columns = [column.name for column in table_class.__table__.columns]

        # Get columns from the SQL database
        table_on_server = self.get_table_on_server(table_dict)
        server_columns = [column.name for column in table_on_server.columns]

        if set(server_columns) != set(local_columns):
            error_msg = f"Columns in table {table_name} do not match between local and SQL server \n" \
                        f"local: {local_columns} \n" \
                        f"server: {server_columns}"
            log_and_print(self.to_log, self.logger, 'error', error_msg)
            raise ValueError(error_msg)

    def add_data_to_server(self, table_dict, data_df):
        data_df.to_sql(
            table_dict['table_name'], self.engine, schema=table_dict['schema'], if_exists='append', index=False
        )

    def update_data_on_server(self, table_dict, pk_value, data_df):
        prime_key = self.table_classes[table_dict['table_name']].__table__.primary_key.columns[0].name

        for index, row in data_df.iterrows():
            data_dict = row.to_dict()
            with self.session_scope() as session:
                session.query(self.table_classes[table_dict['table_name']]).filter_by(
                    **{prime_key: pk_value}
                ).update(data_dict)

    def delete_data_on_server(self, table_dict, pk_value):
        prime_key = self.table_classes[table_dict['table_name']].__table__.primary_key.columns[0].name

        with self.session_scope() as session:
            session.query(self.table_classes[table_dict['table_name']]).filter_by(
                **{prime_key: pk_value}
            ).delete()

    def add_comment_on_server(self, table_dict, pk_value, comment: str):
        self.update_data_on_server(table_dict, pk_value, pd.DataFrame({'comment': [comment]}))

    def add_recipe_data(self, data_df):
        self.add_data_to_server(self.tables_dict['active_learning.recipe'], data_df)

    def fetch_sample_starting_id(self):
        with self.session_scope() as session:
            try:
                sample_id = session.query(self.table_classes['sample'].id).order_by(
                    self.table_classes['sample'].id.desc()
                ).first()[0] + 1
            except TypeError:
                sample_id = 1
        return sample_id

    def fetch_recipe_data_by_arm_name_list(self, arm_name_list, elements: list):
        data_df = None
        for arm_name in arm_name_list:
            if data_df is None:
                data_df = self.fetch_recipe_data_by_arm_name(arm_name, elements)
            else:
                data_df = pd.concat([data_df, self.fetch_recipe_data_by_arm_name(arm_name, elements)], ignore_index=True)
        return data_df

    def fetch_recipe_data_by_arm_name(self, arm_name, elements: list):
        # get all rows with trial_index = trial_index, only show element col in elements list
        col_list = [self.table_classes['recipe'].__table__.columns[element] for element in elements]
        with self.session_scope() as session:
            data_df = pd.read_sql(
                session.query(*col_list).filter_by(arm_name=arm_name).statement, self.engine
            )
        return data_df

    def add_sample_data(self, data_df):
        self.add_data_to_server(self.tables_dict['active_learning.sample'], data_df)

    def fetch_next_sample_batch_id(self):
        with self.session_scope() as session:
            try:
                batch_id = session.query(self.table_classes['sample'].sample_batch_id).order_by(
                    self.table_classes['sample'].sample_batch_id.desc()
                ).first()[0] + 1
            except TypeError:
                batch_id = 1
        return batch_id

    def get_arm_results(self, arm_name, metric_name):
        self.detect_outliers(arm_name, metric_name)
        results = pd.read_sql(
            'SELECT * FROM active_learning.full_table '
            # arm_name has to be bracketed with single quote
            f"WHERE arm_name = \'{arm_name}\' "
            "AND abandoned IS NULL AND outlier IS NULL",
            self.dbConnection
        )
        if len(results) == 0:
            raise ValueError(f'no result matched with arm {arm_name}')
        else:
            return sigfig.round(results.loc[:, metric_name].mean(), 4)

    def detect_outliers(self, arm_name, metric_name):
        results = pd.read_sql(
            'SELECT * FROM active_learning.full_table '
            # arm_name has to be bracketed with single quote
            f"WHERE arm_name = \'{arm_name}\' AND abandoned IS NULL",
            self.dbConnection
        )
        mean = results[metric_name].mean()
        std = results[metric_name].std()
        for i, full_data in results.iterrows():
            if abs(full_data[metric_name] - mean) > 1.96 * std:
                self.set_outlier(full_data['test_id'])

    def set_outlier(self, test_id):
        self.update_data_on_server(
            self.tables_dict['active_learning.performance'],
            test_id,
            pd.DataFrame({'outlier': [1]})
        )

    def create_view(self, elements, metric_name):
        elements_quoted = [f'\"{e}\"' for e in elements]
        recipe_columns = ", ".join(f'r.{e}' for e in elements_quoted)
        self.dbConnection.execute(
            'CREATE VIEW active_learning.full_table AS '
            'SELECT '
            'r.trial_index, '
            'r.arm_index, '
            'r.arm_name, '
            'p.sample_id, '
            's.sample_batch_id, '
            'p.id test_id, '
            f'{recipe_columns}, '
            f'p.\"{metric_name}\", '
            'p.abandoned, '
            'p.abandon_reason, '
            'p.outlier '
            'FROM active_learning.performance p '
            'JOIN active_learning.sample s ON p.sample_id = s.id '
            'JOIN active_learning.recipe r ON s.arm_name = r.arm_name '
        )
        log_and_print(self.to_log, self.logger, 'warning', 'full table view created')
