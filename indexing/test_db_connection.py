import psycopg2


DBNAME = 'vectordb'
USER = "postgres"
PASSWORD = "password"
HOST = "127.0.0.1"
# HOST = "172.17.0.2"
PORT = 5434
TABLE_DOCUMENTS = 'documents'
VECTOR_DIM = 1024

conn = psycopg2.connect(
    database = DBNAME,
    user = USER,
    password = PASSWORD,
    host = HOST,
    port = PORT)
