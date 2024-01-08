# Relational Databases

## Introduction

- What is a database?
  - Organized collection of related data
- Whart is a database management system (DBMS)?
  - Collection of programs that enables users to create and maintain a database
  - allows users to create, query, modify, and manage

**DATABASE != DBMS**

### Reasons for DBMS

- **Efficiency**: Data is stored efficiently.
- **Integrity**: Data is consistent and correct.
- **Security**: Data is safe from unauthorized access.
- **Concurrent Access**: Multiple users can access the same data at the same time.
- **Crash Recovery**: Data is safe from crashes.
- **Independence**: Data is independent of the programs that use it.

## Relational Data Model

- Works with entities and relationships
- **Entity**: a thing or object in the real world that is distinguishable from other objects
  - e.g. students in a school
- **Relationship**: an association among entities
  - e.g. a student is enrolled in a course

## Relational Database

- Collection of relations
- A relation is an instance of a relation schema (similar to object = instance of a class)
- **Relation Schema** specifies:
  1. Name of relation
  2. Name and domain of each attribute
- **Domain**: set of constraints that determines the type, length, format, range, uniqueness and nullability of values stored for an attribute

<img src="images/1_anatomy_table.png" width="400">

## Query language in a DBMS

- Query: a request for information from a database. Results in a relation.
- **Structured Query Language (SQL)**: standard programming language for managing and manipulating databases.
- Can be categorized into:
  - **Data Definition Language (DDL)**: used to define the database structure or schema.
  - **Data Manipulation Language (DML)**: used to read, insert, delete, and modify data.
  - **Data Query Language (DQL)**: used to retrieve data from a database.
  - **Data Control Language (DCL)**: used to control access to data stored in a database.

## PostgreSQL

- Specific flavor of SQL and DBMS
- open-source, cross-platform DBMS that implements the relational model
- reliable, robust, and feature-rich

### psql

| Command | Usage                                      |
| ------- | ------------------------------------------ |
| `\l`    | List all databases                         |
| `\c`    | Connect to a database                      |
| `\d`    | Describe tables and views                  |
| `\dt`   | List tables                                |
| `\dt+`  | List tables with additional info           |
| `\d+`   | List tables and views with additional info |
| `\!`    | Execute shell commands                     |
| `\cd`   | Change directory                           |
| `\i`    | Execute commands from a file               |
| `\h`    | View help on SQL commands                  |
| `\?`    | View help on psql meta commands            |
| `\q`    | Quit interactive shell                     |

### ipython-sql

- To install: `pip install ipython-sql`
- To load: `%load_ext sql`

```python
# Login to database
import json
import urllib.parse

# use credentials.json to login (not included in repo)
with open('data/credentials.json') as f:
    login = json.load(f)

username = login['user']
password = urllib.parse.quote(login['password'])
host = login['host']
port = login['port']
```

- Establish connection:

```
%sql postgresql://{username}:{password}@{host}:{port}/world
```

- Run queries:

```python
output = %sql SELECT name, population FROM country;
```

or

```python
%%sql output << # Not a pandas dataframe
SELECT
  name, population
FROM
  country
;

# convert to pandas dataframe
df = output.DataFrame()
```

- Set configurations:

```python
%config SqlMagic.displaylimit = 20
```

## SQL basic commands

- taken from https://www.sqltutorial.org/sql-cheat-sheet/*

### Querying Data from a table

| Command                                                                        | Description / Example                                |
| ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `SELECT c1, c2`<br>`FROM t;`                                                   | Query data in columns c1, c2 from a table            |
| `SELECT *`<br>`FROM t;`                                                        | Query all rows and columns from a table              |
| `SELECT c1, c2`<br>`FROM t`<br>`WHERE condition;`                              | Query data and filter rows with a condition          |
| `SELECT DISTINCT c1`<br>`FROM t`<br>`WHERE condition;`                         | Query distinct rows from a table                     |
| `SELECT COUNT(DISTINCT (c1, c2))`<br>`FROM t;`                                 | Count distinct rows in a table                       |
| `SELECT c1, c2`<br>`FROM t`<br>`ORDER BY c1 ASC [DESC];`                       | Sort the result set in ascending or descending order |
| `SELECT c1, c2`<br>`FROM t`<br>`ORDER BY c1`<br>`LIMIT n OFFSET offset;`       | Skip offset of rows and return the next n rows       |
| `SELECT c1, aggregate(c2)`<br>`FROM t`<br>`GROUP BY c1;`                       | Group rows using an aggregate function               |
| `SELECT c1, aggregate(c2)`<br>`FROM t`<br>`GROUP BY c1`<br>`HAVING condition;` | Filter groups using HAVING clause                    |
| `SELECT CONCAT(c1, c2)`<br>`FROM t;`                                           | Concatenate two or more strings                      |

**Notes:**

- Conditions: `=`, `>=`, `<=`, `IN ('a','b')`, `IS NULL`, ...
- Strings must be enclosed in single quotes `'...'`
- Aggregate functions: `AVG`, `COUNT`, `MAX`, `MIN`, `SUM`, `ROUND(value, decimal_places)`
- Need decimal point to prevent integer division

### Querying Data from multiple tables

| Command                                                            | Description / Example                         |
| ------------------------------------------------------------------ | --------------------------------------------- |
| `SELECT c1, c2`<br>`FROM t1`<br>`INNER JOIN t2 ON condition;`      | Inner join t1 and t2                          |
| `SELECT c1, c2`<br>`FROM t1`<br>`LEFT JOIN t2 ON condition;`       | Left join t1 and t2                           |
| `SELECT c1, c2`<br>`FROM t1`<br>`RIGHT JOIN t2 ON condition;`      | Right join t1 and t2                          |
| `SELECT c1, c2`<br>`FROM t1`<br>`FULL OUTER JOIN t2 ON condition;` | Perform full outer join                       |
| `SELECT c1, c2`<br>`FROM t1`<br>`CROSS JOIN t2;`                   | Produce a Cartesian product of rows in tables |
| `SELECT c1, c2`<br>`FROM t1 A`<br>`INNER JOIN t2 B ON condition;`  | Join t1 to itself using INNER JOIN clause     |

### Using SQL Operators

| Command                                                                   | Description / Example                         |
| ------------------------------------------------------------------------- | --------------------------------------------- |
| `SELECT c1, c2`<br>`FROM t1`<br>`UNION [ALL]`<br>`SELECT c1, c2 FROM t2;` | Combine rows from two queries                 |
| `SELECT c1, c2`<br>`FROM t1`<br>`INTERSECT`<br>`SELECT c1, c2 FROM t2;`   | Return the intersection of two queries        |
| `SELECT c1, c2`<br>`FROM t1`<br>`MINUS`<br>`SELECT c1, c2 FROM t2;`       | Subtract a result set from another result set |
| `SELECT c1, c2`<br>`FROM t`<br>`WHERE c1 [NOT] LIKE pattern;`             | Query rows using pattern matching % \_        |
| `SELECT c1, c2`<br>`FROM t`<br>`WHERE c1 [NOT] IN value_list;`            | Query rows in a list                          |
| `SELECT c1, c2`<br>`FROM t`<br>`WHERE c1 BETWEEN low AND high;`           | Query rows between two values                 |
| `SELECT c1, c2`<br>`FROM t`<br>`WHERE c1 IS [NOT] NULL;`                  | Check if values in a table is NULL or not     |

### Managing Tables

| Command                                                                                            | Description / Example                 |
| -------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `CREATE TABLE t (`<br>`id INT PRIMARY KEY,`<br>`name VARCHAR NOT NULL,`<br>`price INT DEFAULT 0);` | Create a new table with three columns |
| `DROP TABLE t;`                                                                                    | Delete the table from the database    |
| `ALTER TABLE t ADD column;`                                                                        | Add a new column to the table         |
| `ALTER TABLE t DROP COLUMN c;`                                                                     | Drop column c from the                |

## Database Data Types

PostgreSQL supports many data types. The most common ones are:

- **Boolean**, `BOOLEAN` or `BOOL`
  - True: `TRUE`, `t`, `true`, `y`, `yes`, `on`, `1`
- **Characters**
  - `CHAR(n)`: string of `n` characters (pad with spaces)
  - `VARCHAR(n)`: string of up to `n` characters
  - `TEXT`: PostgreSQL-specific type for storing strings of any length
- **DateTime**: date and time
  - `DATE`: date (YYYY-MM-DD)
    - `CURRENT_DATE`: current date
  - `TIME`: time
  - `TIMESTAMP`: date and time
  - `TIMESTAMPTZ`: date and time with timezone
- **Binary**: binary data
- **Numbers**

## " vs '

- `"` is used for identifiers (e.g. table names, column names)
- `'` is used for strings.

### Numbers

| Name        | Storage Size | Description                     | Range                                                    |
| ----------- | ------------ | ------------------------------- | -------------------------------------------------------- |
| `smallint`  | 2 bytes      | small-range integer             | -32,768 to +32,767                                       |
| `integer`   | 4 bytes      | typical choice for integer      | -2,147,483,648 to +2,147,483,647                         |
| `bigint`    | 8 bytes      | large-range integer             | -9,223,372,036,854,775,808 to +9,223,372,036,854,775,807 |
| `serial`    | 4 bytes      | auto-incrementing integer       | 1 to 2,147,483,647                                       |
| `bigserial` | 8 bytes      | large auto-incrementing integer | 1 to 9,223,372,036,854,775,807                           |

_Note: `serial` and `bigserial` are not true types, but merely a notational convenience for creating unique identifier columns._

**Floating-Point Numbers**

| Name               | Storage Size | Description                 | Range                                                 |
| ------------------ | ------------ | --------------------------- | ----------------------------------------------------- |
| `real`             | 4 bytes      | variable-precision, inexact | at least 6 decimal digits (implementation dependent)  |
| `double precision` | 8 bytes      | variable-precision, inexact | at least 15 decimal digits (implementation dependent) |

**Arbitrary Precision Numbers**

| Name                   | Storage Size | Description                     | Range                                                         |
| ---------------------- | ------------ | ------------------------------- | ------------------------------------------------------------- |
| `numeric` or `decimal` | variable     | user-specified precision, exact | 131072 digits before and 16383 digits after the decimal point |

_Note: `numeric` and `decimal` are the same type in Postgres._

### Type conversion

- Can do : `<col>::<data_type>` or `CAST(<col> AS <data_type>)`
- e.g. `SELECT '123'::integer;`
- e.g. `SELECT CAST('123' AS integer);`

### Filtering using `WHERE`

```sql
SELECT *
FROM table_name
WHERE [NOT]
    condition
    AND/OR
    condition;
```

| Condition Example                             | Description                                                  |
| --------------------------------------------- | ------------------------------------------------------------ |
| `WHERE column = value`                        | Equals; returns true if the column equals the value.         |
| `WHERE column <> value`                       | Not equals; true if the column is not equal to value.        |
| `WHERE column > value`                        | Greater than; true if the column is more than value.         |
| `WHERE column < value`                        | Less than; true if the column is less than value.            |
| `WHERE column BETWEEN value1 AND value2`      | True if the column is within the range of value1 and value2. |
| `WHERE column [NOT] IN (value1, value2, ...)` | True if the column is equal to any of multiple values.       |
| `WHERE column [NOT] LIKE pattern`             | True if the column matches the SQL pattern.                  |
| `WHERE column IS [NOT] NULL`                  | True if the column is NULL.                                  |

#### Pattern Matching using `LIKE`

- `LIKE` is case-sensitive
- `ILIKE` is case-insensitive
- Wildcards:
  - `%` (any string of zero or more characters)
  - `_` (any single character)
- e.g. `WHERE column LIKE 'abc%'`

Other nuances:

- using `ESCAPE` to identify escape character
  - e.g. `WHERE column LIKE '%$%%' ESCAPE '$'`: matches strings that contains `%`

### Column Aliases using `AS`

```sql
SELECT column AS alias
FROM table_name;
```

- Cannot use alias in `WHERE` clause because order of execution is `FROM` then `WHERE`

### ORDER OF EXECUTION

1. `FROM` and `JOIN`
2. `WHERE`
3. `GROUP BY`, then `HAVING`
4. `SELECT` , then `DISTINCT`
5. `ORDER BY`
6. `LIMIT` and `OFFSET`

### Functions and Operators

#### Math

| Short Description            | Example/Syntax     |
| ---------------------------- | ------------------ |
| Addition                     | `col1 + col2`      |
| Subtraction                  | `col1 - col2`      |
| Multiplication               | `col1 * col2`      |
| Division                     | `col1 / col2`      |
| Modulus                      | `col1 % col2`      |
| Absolute Value               | `ABS(col)`         |
| Round to n decimal places    | `ROUND(col, n)`    |
| Round up                     | `CEILING(col)`     |
| Round down                   | `FLOOR(col)`       |
| Power of n                   | `POWER(col, n)`    |
| Square Root                  | `SQRT(col)`        |
| Truncate to n decimal places | `TRUNCATE(col, n)` |
| Generate random number       | `RAND()`           |

#### String

| Short Description     | Example/Syntax                                |
| --------------------- | --------------------------------------------- |
| Concatenate strings   | `CONCAT(str1, str2, ...)` or `str1 \|\| str2` |
| Length of string      | `CHAR_LENGTH(str)`                            |
| Convert to lower case | `LOWER(str)`                                  |
| Convert to upper case | `UPPER(str)`                                  |
| Extract substring     | `SUBSTRING(str, start, length)`               |
| Trim spaces           | `TRIM(str)`                                   |
| Replace substring     | `REPLACE(str, from_str, to_str)`              |
| Position of substring | `POSITION(substring IN str)`                  |

#### Datetime

| Short Description                | Example/Syntax                 |
| -------------------------------- | ------------------------------ |
| Current date                     | `CURRENT_DATE`                 |
| Current time                     | `CURRENT_TIME`                 |
| Current date and time            | `CURRENT_TIMESTAMP`            |
| Extract part of date/time        | `EXTRACT(part FROM date/time)` |
| Add interval to date/time        | `date/time + INTERVAL`         |
| Subtract interval from date/time | `date/time - INTERVAL`         |
| Difference between dates/times   | `DATEDIFF(date1, date2)`       |
| Format date/time                 | `FORMAT(date/time, format)`    |

e.g. `SELECT EXTRACT(YEAR FROM CURRENT_DATE);`

#### Null

| Short Description                 | Example/Syntax                                            |
| --------------------------------- | --------------------------------------------------------- |
| Check for NULL                    | `col IS NULL`                                             |
| Check for non-NULL                | `col IS NOT NULL`                                         |
| Replace NULL with specified value | `COALESCE(col, replace_value)`                            |
| Null-safe equal to operator       | `col1 <=> col2`                                           |
| Case statement with NULL handling | `CASE WHEN col IS NULL THEN result ELSE other_result END` |
| Null if expression is NULL        | `NULLIF(expression, NULL)`                                |

## Aggregate Functions

| Function  | Description                                               |
| --------- | --------------------------------------------------------- |
| `AVG()`   | Returns the average value.                                |
| `COUNT()` | Returns the number of rows.                               |
| `MAX()`   | Returns the maximum value.                                |
| `MIN()`   | Returns the minimum value.                                |
| `SUM()`   | Returns the sum of all or distinct values.                |
| `VAR()`   | Returns the variance of all or distinct values.           |
| `STDDEV`  | Returns the standard deviation of all or distinct values. |

**Important notes:**

- Cannot use aggregate function with normal column (in `SELECT` clause) without `GROUP BY` clause
- All aggregate functions ignore `NULL` values except `COUNT(*)`
- Cannot use aggregate function in `WHERE` clause because order of execution is `FROM` then `WHERE`

## Grouping

- `GROUP BY` clause is used to group rows with the same values

```sql
-- Formal syntax
SELECT
    grouping_columns, aggregated_columns
FROM
    table1
WHERE -- filter rows before grouping
    condition
GROUP BY -- must be between WHERE and ORDER BY
    grouping_columns
HAVING -- Used to filter groups (after grouping)
    group_condition
ORDER BY
    grouping_columns
```

## Joining Tables

- `JOIN` clause is used to combine rows from two or more tables based on a related column between them
  - DEFAULT: `INNER JOIN`

```sql
SELECT -- columns from both tables
    t1.column1, t2.column2
FROM
    table1 AS t1 -- after using alias, use alias instead of table name
jointype
    table2 AS t2
ON -- condition to join tables
    t1.column = t2.column
```

| Join Type     | Description                                                                                                      |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `CROSS`       | Returns the Cartesian product of the sets of rows from the joined tables (all possible combinations)             |
| `INNER`       | Returns rows when there is a match in both tables. (intersect)                                                   |
| `NATURAL`     | Returns all rows without specifying ON, names have to be the same in both tables.                                |
| `LEFT OUTER`  | Returns all rows from the left-hand table, plus any rows in the right-hand table that match the left-hand side.  |
| `RIGHT OUTER` | Returns all rows from the right-hand table, plus any rows in the left-hand table that match the right-hand side. |
| `FULL OUTER`  | Returns all rows from both tables, with nulls in place of those rows that have no match in the other table.      |

## Data Manipulation

### INSERT

- Add new rows to table, by:
  - column position:
    ```sql
        INSERT INTO table_name
        VALUES
            (value1, value2, ...),
            (value1, value2, ...), -- can insert multiple rows at once
            ...
    ```
    - values need to be in the same order as the columns
    - don't need to specify column names
  - column name:
    ```sql
    INSERT INTO table_name(column1, column2, ...)
    VALUES (value1, value2, ...)
    ```
    - values can be in any order
    - need to specify column names
  - from another table:
    ```sql
    INSERT INTO
        table_name(column1, column2, ...)
    SELECT *
    FROM other_table
    ```

### UPDATE

- Modify existing rows in table

```sql
UPDATE table_name
SET column1 = value1,
    column2 = value2,
    ...
WHERE condition
```

### DELETE

- Remove rows from table
- removes all rows but keeps table if no `WHERE` clause

```sql
DELETE FROM table_name
WHERE condition
```

### TRUNCATE

- Remove all rows from table
- faster than `DELETE` because it doesn't scan every row
  - also does not log each row deletion

```sql
TRUNCATE TABLE table_name
```

## Creating, Altering, and Dropping Tables

### CREATE TABLE

```sql
CREATE TABLE table_name (
    column1 datatype [constraint] PRIMARY KEY, -- for simple primary key
    column2 datatype UNIQUE, -- unique constraint
    column3 TEXT DEFAULT 'default value', -- default value
    column4 datatype
        [CONSTRAINT constraint_name] CHECK (condition), -- check constraint
    -- e.g.
    name TEXT NOT NULL,
    phone CHAR(12) CHECK (phone LIKE '___-___-____')
    ...

    -- table constraints
    -- if set composite primary key, also can simple
    [CONSTRAINT constraint_name] PRIMARY KEY (column1, column2, ...),
);
```

### Keys

- **Simple key**: a single column
- **Composite key**: multiple columns

#### Candidate Key

- Can uniquely identify a table row
- Must be **minimal**: no subset of the candidate key can be a candidate key
- Can have multiple in a table (e.g. `id` and `email`)

#### Primary Key

- A candidate key that is chosen to be the main identifier of a table
- Automaticaly **unique** and **not null**
- Must be unique and not null
- Must be **minimal**
  - generally the candidate key with the fewest columns

#### Foreign Key

- A column that references a primary key in another table
- Prevents invalid data from being inserted
- Can be null
- **Child table**: table with foreign key
- **Parent table**: table with primary key

```sql
-- parent table
CREATE TABLE instructor (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
);

-- child table
CREATE TABLE instructor_course (
    id INTEGER,
    course_id TEXT,
    PRIMARY KEY (id, course_id), -- composite primary key
    FOREIGN KEY (id) REFERENCES instructor (id)
        ON DELETE CASCADE,

    -- can also specify column name
    FOREIGN KEY (course_id) REFERENCES instructor (id)
        ON DELETE CASCADE -- delete child rows when parent row is deleted
        ON UPDATE CASCADE -- update child rows when child row is updated
);
```

- Foreign key constraint ensures there are no orphaned rows
  - i.e. all foreign key in child table must exist in parent table

| Referential Action | Description                                                         |
| ------------------ | ------------------------------------------------------------------- |
| `NO ACTION`        | Default. Rejects update or delete of parent row                     |
| `SET NULL`         | Sets foreign key to null                                            |
| `CASCADE`          | Deletes or updates child rows when parent row is deleted or updated |
| `SET DEFAULT`      | Sets foreign key to default value                                   |

### TEMPORARY TABLE

- Used to store temporary data
- Automatically dropped at end of session
- Private to current session
- Physically stored in temp tablespace

```sql
CREATE TEMPORARY TABLE table_name (
    ...
);

-- can also create from another table
CREATE TEMPORARY TABLE
    temp_table_name
AS
    SELECT name, department FROM other_table
;
```

## DROP TABLE

- Remove table and all its data
- Postgres does not allow dropping a table if it has a foreign key constraint
  - need to drop the foreign key constraint first
  - or use `CASCADE` to drop the table and all its foreign key constraints

```sql
DROP TABLE table_name [CASCADE];
```

## Transaction

- A transaction is a sequence of SQL statements that are treated as a unit, so either all of them are executed, or none of them are executed.

```sql
[{BEGIN|START} [TRANSACTION]]; -- BEGIN more common, can just start  with `BEGIN`

<SQL statements>

{COMMIT|ROLLBACK};
```

- `COMMIT`: make all changes made by the transaction permanent
  - without COMMIT, changes are not visible to other users
- `ROLLBACK`: undo all changes made by the transaction

### ACID

Properties of a transaction:

- **Atomicity**: all or nothing (transaction either completes or is aborted)
- **Consistency**: database must be in a consistent state before and after the transaction
  - e.g. no scores > 100, transaction will not be committed if it violates this
- **Isolation**: if two transactions are executed concurrently, the result should be the same as if they were executed one after the other
- **Durability**: changes made by a committed transaction must be permanent (even if there is a system failure), achieved using transaction log

## Subqueries

- A subquery is a query within a query
- Normally used to mix aggregate and non-aggregate queries
- Can only be used if subquery **returns ONLY 1 value**.

```sql
-- Selects countries with a population greater than the average population of all countries
SELECT
    name
FROM
    country
WHERE
    surfacearea > (
        SELECT AVG(surfacearea) FROM country
    )
;
```

### Correlated Subqueries

- A correlated subquery is a subquery that uses values from the outer query

```sql
-- Selects countries with the largest population in their continent
SELECT
    c1.name, c1.continent
FROM
    country c1
WHERE
    c1.population = (
        SELECT
            MAX(c2.population)
        FROM
            country c2
        WHERE
            c2.continent = c1.continent
    )
;
```

### ANY, ALL, EXISTS

- `ANY` and `ALL` are used to compare a value to a list or subquery

| ANY                           | ALL                        |
| ----------------------------- | -------------------------- |
| if one of values true => true | if all values true => true |

e.g.

```sql
-- Find the names of all countries that have a
-- population greater than all European countries.
SELECT
    name
FROM
    country
WHERE
    continent <> 'Europe'
    AND
    population > ALL (
        SELECT
            population
        FROM
            country
        WHERE
            continent = 'Europe'
    )
;
```

- `EXISTS` is used to check if a subquery returns any rows
  - faster than `IN` because it stops as soon as it finds a match

````sql
SELECT co.name
FROM country co
WHERE EXISTS (SELECT * FROM city ci
        WHERE co.code = ci.countrycode AND ci.population > 5000000);
        ```
````

## Views

- Is a named _result_ of a `SELECT` query
  - Behaves like a table (called _virtual table_ sometimes)
  - Current and Dynamic: whenever you query a view, you get the most up-to-date data
  - Views persists: they are kept
  - Not materialized: they are not stored in the database
  - Views can hide complexity
  - Manage access to data
  - Do not support constraints

```sql
CREATE VIEW view_name AS
    select_statment
;

DROP VIEW [IF EXISTS] view_name;
```

### Materialized Views

- Materialized views are stored in the database
- They are updated periodically
- They are used for performance reasons (a lot faster)

```sql
CREATE MATERIALIZED VIEW my_mat_view AS
    select_statement
;

REFRESH MATERIALIZED VIEW [CONCURRENTLY] my_mat_view;
-- CONCURRENTLY: allows you to query the view while it is being refreshed, no guarantee it is up-to-date

DROP MATERIALIZED VIEW my_mat_view;
```

## Temporary Tables

```sql
DROP TABLE IF EXISTS temp_table_name;

CREATE TEMPORARY TABLE temp_table_name AS (
    ...
);
```

## CTEs

- Common Table Expressions: temporary named result of a query

```sql
WITH
    expression_name [(column_names, ...)]
AS (
    query
)
query
-- SELECT * FROM expression_name;
-- This query needs to use the column names defined in the CTE after the expression name
;
```

## Summary TT vs Views vs CTEs

| Feature          | Temporary Tables                              | Views                                                | Materialized Views                          | CTEs                                                                                                           |
| ---------------- | --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Nature**       | Physical table, session-scoped                | Virtual table                                        | Physical storage of query results           | Temporary result set                                                                                           |
| **Data Storage** | Stores data temporarily                       | Does not store data, only stores the query itself    | Stores results of a query                   | Does not store data, used within query execution                                                               |
| **Scope**        | Exists only during a database session         | Permanent, unless dropped                            | Permanent, unless dropped                   | Exists only during the execution of a query                                                                    |
| **Access**       | Can only be accessed by 1 user                | Can be accessed by multiple users                    | Can be accessed by multiple users           | Can be accessed by multiple users                                                                              |
| **Usage**        | Intermediate data storage, complex queries    | Simplifying access to complex queries, data security | Performance improvement for complex queries | Breaking down complex queries, recursive queries <br /> can manipulate default processing order of SQL clauses |
| **Performance**  | Dependent on data size and indexes            | Executes underlying query each time                  | Faster access, as data is pre-calculated    | Dependent on the complexity of the query                                                                       |
| **Updates**      | Data persists for the session, can be updated | Reflects real-time data from base tables             | Requires refresh to update data             | Not applicable (recomputed with each query execution)                                                          |
| **Indexing**     | Can have indexes                              | Cannot have indexes                                  | Can have indexes                            | Not applicable                                                                                                 |

## Window Functions

- Window functions are used to compute aggregate values over a group of rows
- **Only allowed in `SELECT` and `ORDER BY` clauses**
- Processed:
  - **after** `WHERE`, `GROUP BY`, and `HAVING`
  - **before** `SELECT` and `ORDER BY`

```sql
SELECT
    column,
    window_function_name(expression) OVER (
        PARTITION BY column
        ORDER BY column
        frame_clause
    )
    -- e.g.
    continent,
    MAX(population)
        OVER (PARTITION BY continent)
FROM
    table
;
```

### Window Function Types

- Aggregate functions
  - `AVG`, `COUNT`, `MAX`, `MIN`, `SUM`
- Ranking functions
  - `CUME_DIST()`: returns the cumulative distribution, i.e. the percentage of values less than or equal to the current value.
  - `NTILE()`: given a specified number of buckets, it tells us in which bucket each row goes among other rows in the partition.
  - `PERCENT_RANK()`: similar to CUME_DIST(), but considers only the percentage of values less than (and not equal to) the current value.
  - `DENSE_RANK()`: returns the rank of a row within a partition without jumps after duplicate ranks (e.g. 1, 2, 2, 3, …)
  - `RANK()`: returns the rank of row within a partition but with jumps after duplicates ranks (e.g. 1, 2, 2, 4, …)
  - `ROW_NUMBER()`: returns simply the number of a row in a partition, regardless of duplicate values

## Relational vs. Non-relational Databases

| Elements    | Relational                                               | Non-relational                                                   |
| ----------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| Data Model  | tables                                                   | key-value pairs, documents, graphs, etc.                         |
| Schema      | fixed (updates are complicated and time-consuming)       | dynamic (schema-free)                                            |
| Querying    | semi standard (SQL)                                      | proprietary                                                      |
| Performance | strong data consistency and integrity                    | faster performance for specific usecases                         |
| Efficiency  | write times and table locks reduce efficiency            | faster read and write times                                      |
| Scalability | vertical scaling                                         | horizontal scaling                                               |
| Development | require more effort                                      | easier to develop and require fewer resources                    |
| Principles  | **ACID** (Atomicity, Consistency, Isolation, Durability) | **BASE** (Basically Available, Soft state, Eventual consistency) |

## NoSQL (Not Only SQL)

- **Main goals**:
  - reduce the dependence on fixed schema
  - reduce the dependence on a single point of truth
  - scaling

### Strengths

- Schema-free
- Accepting Basic Availability
  - usually see all tweets, but sometimes a refresh shows new tweets
- BASE:
  - Basically Available
  - Soft state
  - Eventual consistent

### Types of NoSQL Databases

- **Key-value stores**: like a dictionary
  - Redis, Memcached, Amazon DynamoDB
- **Document stores**: type of key-value store but with an internal searchable structure.
  - Document: contains all the relevant data and has a unique key
  - MongoDB, CouchDB
- **Column stores**: based on Google's BigTable
  - Cassandra, HBase, Google BigTable
- **Graph databases**: based on graph theory
  - Neo4j, OrientDB, Amazon Neptune

## MongoDB

- Based on _JSON-like_ (JavaScript Object Notation) documents for storage:
  - Actually it is **BSON** (Binary JSON)
  - Max BSON document size: 16MB
- Structure:
  - **Database**: `client.get_database_names()`
  - **Collection**: `client[db_name].get_collection_names()`
  - **Document**: `client[db_name][collection_name].find_one()` or `client[db_name][collection_name].find()`

### `find`

- `client[db_name][collection_name].find()`
  - returns a cursor
  - `list(client[db_name][collection_name].find())` returns a list of documents
  - `next(client[db_name][collection_name].find())` returns the next document

```python
client = MongoClient()
client[db_name][collection_name].find(
    filter={...},
    projection={...},
    sort=[...],
    skip=...,
    limit=...,
)
```

#### `filter`

- `filter` is a dictionary
- `filter = {"key": "value"}` returns all documents where `key` is `value`

| Operator        | symbol      | notes                                                                                   |
| --------------- | ----------- | --------------------------------------------------------------------------------------- |
| `$eq` or `$neq` | `=` or `~=` |
| `$gt` or `$gte` | `>` or `>=` |
| `$lt` or `$lte` | `<` or `<=` |
| `$ne`           | `!=`        |
| `$in`           | `in`        | if any of the values in the list matches the value of the key, the document is returned |
| `$nin`          | `not in`    |
| `$all`          | `all`       | all values in the list must match the value of the keys                                 |
| `$exists`       | `exists`    |
| `$regex`        | `regex`     |

- Also can use `$not` to negate the operator

* using `$or`, `$nor` and `$and`:

```python
filter = {
    "$or": [
        {"key_1": "value_1"},
        {"key_2": "value_2"},
    ],
    "$and": [
        {"key_3": "value_3"},
        {"key_4": "value_4"},
    ],
    "key_5": {"$gte": 10, "$lte": 20}, # and is implied
}

filter = {
  "key_1": "value_1",
  "key_2": "value_2", # and is implied
}
```

#### `projection`

- `projection` is a dictionary
- `projection = {"key_1": 1, "key_2": 0}` returns all documents with only `key_1` and without `key_2`

#### `sort`

- `sort` is a list of tuples
  - 1: ascending order
  - -1: descending order
- `sort = [("key_1", 1), ("key_2", -1)]` sorts by `key_1` in ascending order and then by `key_2` in descending order

#### `limit`

- `limit` is an integer
- `limit = 10` returns the first 10 documents

#### `skip`

- `skip` is an integer
- `skip = 10` skips the first 10 documents

#### `count_documents`

- `count_documents` is a method
- `client[db_name][collection_name].count_documents(filter={...})` returns the number of documents that match the filter

#### `distinct`

- `distinct` is a method
- `client[db_name][collection_name].distinct("key_1")` returns a list of distinct values for `key_1`
