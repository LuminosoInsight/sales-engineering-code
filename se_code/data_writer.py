import csv
import psycopg2
import psycopg2.extras
from psycopg2 import Error


class LumiDataWriter:
    def __init__(self, project_id):
        self.project_id = project_id

    def output_data(self, data):
        print("LumiDataWriter output_data. Subclass should have handled this")


class LumiCsvWriter(LumiDataWriter):
    def __init__(self, filename, table_name, project_id, encoding):
        super().__init__(project_id)

        self.filename = filename
        self.table_name = table_name
        self.encoding = encoding
        self.file = open(filename, 'w', encoding=encoding, newline='')

    def output_data(self, data):
        if len(data) == 0:
            print('Warning: No data to write to {}.'.format(self.filename))
            return
        # Get the names of all the fields in all the dictionaries in the table.  We
        # want a set rather then a list--but Python sets don't respect ordering,
        # and we want to keep the columns in the same order as much as possible,
        # so we put them into a dictionary with dummy values.
        fieldnames = {k: None for t_item in data for k in t_item}
        writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


class LumiSqlWriter(LumiDataWriter):
    def __init__(self, sql_connection, table_schemas, table_name, project_id):
        super().__init__(project_id)
        self.table_schemas = table_schemas
        self.sql_connection = sql_connection
        self.table_name = table_name

    def output_data(self, data):
        # every table has a project_id, but most data doesn't have the column
        # just add it here. Aware that this modifies the data
        # for later calls, but this data is all transient and
        # only for output anyway
        for r in data:
            r['project_id'] = self.project_id

        if len(data) > 0:
            keys = list(set(val for dic in data for val in dic.keys()))
            columns = ', '.join(keys)

            sql_data = []
            for row in data:
                tup = ()

                for k in keys:
                    if k in row:
                        val = row[k]
                        # k=col_name val=(col_name, col_type, max_len)
                        if self.table_schemas[self.table_name][k][1] in ["varchar", "text"]:
                            val = str(val)
                            if self.table_schemas[self.table_name][k][2] > 0:
                                val = val[0:self.table_schemas[self.table_name][k][2]]
                        tup += (str(val),)
                    else:
                        if self.table_schemas[self.table_name][k][1] in ["numeric"]:
                            tup += (0.0,)
                        else:
                            tup += ("",)

                sql_data.append(tup)

            cursor = self.sql_connection.cursor()
            insert_query = f"INSERT INTO {self.table_name} ({columns}) VALUES %s"
            psycopg2.extras.execute_values(
                cursor, insert_query, sql_data, template=None, page_size=100
            )

            self.sql_connection.commit()
            cursor.close()
